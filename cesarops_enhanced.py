#!/usr/bin/env python3
"""
CESAROPS - Civilian Emergency SAR Operations
Enhanced version with improved reliability, offline capabilities, and ML integration
"""

import os
import sys
import json
import sqlite3
import logging
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue

# Core dependencies
import pandas as pd
import numpy as np
import requests
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
import yaml

# Optional ML dependencies (graceful degradation if not available)
try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML libraries not available - running in basic mode")

# KML support
try:
    import simplekml
    KML_AVAILABLE = True
except ImportError:
    KML_AVAILABLE = False
    print("KML support not available")

class CESAROPSError(Exception):
    """Custom exception for CESAROPS operations"""
    pass

class Logger:
    """Centralized logging system"""
    def __init__(self, log_file="cesarops.log"):
        self.log_file = log_file
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def warning(self, msg): self.logger.warning(msg)
    def debug(self, msg): self.logger.debug(msg)

class DataManager:
    """Handles local data storage and retrieval"""
    def __init__(self, db_path="cesarops_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for caching"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Current data cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                lat REAL,
                lon REAL,
                u REAL,
                v REAL,
                source TEXT,
                fetch_time TEXT
            )
        ''')
        
        # Historical tracks for ML training
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seed_id TEXT,
                step INTEGER,
                timestamp TEXT,
                lat REAL,
                lon REAL,
                predicted_lat REAL,
                predicted_lon REAL,
                actual_lat REAL,
                actual_lon REAL
            )
        ''')
        
        # Model performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT,
                accuracy_score REAL,
                mae REAL,
                rmse REAL,
                test_date TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def cache_current_data(self, df: pd.DataFrame, source: str):
        """Cache current data to database"""
        conn = sqlite3.connect(self.db_path)
        df_copy = df.copy()
        df_copy['source'] = source
        df_copy['fetch_time'] = datetime.now(timezone.utc).isoformat()
        df_copy.to_sql('current_data', conn, if_exists='append', index=False)
        conn.close()
    
    def get_cached_data(self, hours_old=24) -> pd.DataFrame:
        """Retrieve cached current data"""
        conn = sqlite3.connect(self.db_path)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_old)).isoformat()
        
        query = """
        SELECT timestamp, lat, lon, u, v, source 
        FROM current_data 
        WHERE fetch_time > ? 
        ORDER BY fetch_time DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff,))
        conn.close()
        return df

class ERDDAPClient:
    """Enhanced ERDDAP client with retry logic and caching"""
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 5):
        self.base = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        # Add headers to appear more like a regular browser
        self.session.headers.update({
            'User-Agent': 'CESAROPS/1.0 SAR-Tool (Python/requests)'
        })
    
    def _request_with_retry(self, url: str, params: Dict = None) -> requests.Response:
        """Make request with exponential backoff retry"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise CESAROPSError(f"Failed to fetch data after {self.max_retries} attempts: {e}")
                # Exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                Logger().warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                threading.Event().wait(wait_time)
    
    def search_datasets(self, keywords: List[str]) -> List[Dict]:
        """Search for datasets containing keywords"""
        query = ' '.join(keywords)
        url = f"{self.base}/search/index.json"
        
        response = self._request_with_retry(url, {'searchFor': query, 'itemsPerPage': 100})
        data = response.json()
        
        datasets = []
        if 'table' in data and 'rows' in data['table']:
            columns = data['table']['columnNames']
            for row in data['table']['rows']:
                dataset = dict(zip(columns, row))
                datasets.append(dataset)
        
        return datasets
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get detailed information about a dataset"""
        url = f"{self.base}/info/{dataset_id}/index.json"
        response = self._request_with_retry(url)
        return response.json()
    
    def fetch_tabular_data(self, dataset_id: str, variables: List[str], 
                          constraints: List[str] = None) -> pd.DataFrame:
        """Fetch tabular data with constraints"""
        var_string = ','.join(variables)
        url = f"{self.base}/tabledap/{dataset_id}.csv"
        
        params = {'query': var_string}
        if constraints:
            for constraint in constraints:
                params['query'] += f"&{constraint}"
        
        response = self._request_with_retry(url, params)
        
        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        return df

class CurrentsFetcher:
    """Enhanced currents data fetcher with multiple sources"""
    
    KNOWN_SOURCES = {
        'LMHOFS': {
            'base_url': 'https://coastwatch.glerl.noaa.gov/erddap',
            'keywords': ['LMHOFS', 'Lake Michigan', 'surface', 'currents'],
            'variables': ['time', 'latitude', 'longitude', 'u', 'v']
        },
        'RTOFS': {
            'base_url': 'https://coastwatch.pfeg.noaa.gov/erddap',
            'keywords': ['RTOFS', 'ocean', 'currents'],
            'variables': ['time', 'lat', 'lon', 'u_velocity', 'v_velocity']
        },
        'HYCOM': {
            'base_url': 'https://tds.hycom.org/erddap',
            'keywords': ['HYCOM', 'surface', 'currents'],
            'variables': ['time', 'latitude', 'longitude', 'water_u', 'water_v']
        }
    }
    
    def __init__(self, data_manager: DataManager, logger: Logger):
        self.data_manager = data_manager
        self.logger = logger
    
    def auto_discover_dataset(self, source_name: str) -> Optional[str]:
        """Automatically discover appropriate dataset for a source"""
        if source_name not in self.KNOWN_SOURCES:
            raise CESAROPSError(f"Unknown source: {source_name}")
        
        source_config = self.KNOWN_SOURCES[source_name]
        client = ERDDAPClient(source_config['base_url'])
        
        try:
            datasets = client.search_datasets(source_config['keywords'])
            
            # Find dataset with required variables
            for dataset in datasets:
                try:
                    info = client.get_dataset_info(dataset['datasetID'])
                    variables = set()
                    if 'table' in info and 'rows' in info['table']:
                        for row in info['table']['rows']:
                            if len(row) > 1 and row[0] == 'variable':
                                variables.add(row[1])
                    
                    # Check if dataset has required variables (flexible matching)
                    required = {'time', 'lat', 'lon', 'u', 'v'}
                    if any(var in variables for var in ['latitude', 'lat']):
                        if any(var in variables for var in ['longitude', 'lon']):
                            if any(var in variables for var in ['u', 'u_velocity', 'water_u']):
                                if any(var in variables for var in ['v', 'v_velocity', 'water_v']):
                                    self.logger.info(f"Found suitable dataset: {dataset['datasetID']}")
                                    return dataset['datasetID']
                
                except Exception as e:
                    self.logger.debug(f"Error checking dataset {dataset.get('datasetID', 'unknown')}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error discovering dataset for {source_name}: {e}")
        
        return None
    
    def fetch_currents(self, source_name: str, bbox: Tuple[float, float, float, float],
                      start_time: str, end_time: str, dataset_id: str = None) -> pd.DataFrame:
        """Fetch current data from specified source"""
        
        if source_name not in self.KNOWN_SOURCES:
            raise CESAROPSError(f"Unknown source: {source_name}")
        
        source_config = self.KNOWN_SOURCES[source_name]
        
        if not dataset_id:
            dataset_id = self.auto_discover_dataset(source_name)
            if not dataset_id:
                raise CESAROPSError(f"Could not find suitable dataset for {source_name}")
        
        client = ERDDAPClient(source_config['base_url'])
        
        # Build constraints
        min_lon, max_lon, min_lat, max_lat = bbox
        constraints = [
            f"time>={start_time}",
            f"time<={end_time}",
            f"latitude>={min_lat}",
            f"latitude<={max_lat}",
            f"longitude>={min_lon}",
            f"longitude<={max_lon}"
        ]
        
        try:
            df = client.fetch_tabular_data(dataset_id, source_config['variables'], constraints)
            
            # Standardize column names
            column_mapping = {
                'latitude': 'lat', 'lat': 'lat',
                'longitude': 'lon', 'lon': 'lon',
                'u_velocity': 'u', 'water_u': 'u', 'u': 'u',
                'v_velocity': 'v', 'water_v': 'v', 'v': 'v',
                'time': 'timestamp'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure we have required columns
            required_cols = ['timestamp', 'lat', 'lon', 'u', 'v']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise CESAROPSError(f"Missing required columns: {missing}")
            
            # Cache the data
            self.data_manager.cache_current_data(df, source_name)
            
            self.logger.info(f"Fetched {len(df)} current records from {source_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching currents from {source_name}: {e}")
            # Try to return cached data as fallback
            cached = self.data_manager.get_cached_data(hours_old=48)
            if not cached.empty:
                self.logger.warning("Using cached data as fallback")
                return cached
            raise

class MLEnhancedDriftModel:
    """Machine Learning enhanced drift prediction"""
    
    def __init__(self, data_manager: DataManager, logger: Logger):
        self.data_manager = data_manager
        self.logger = logger
        self.model_u = None
        self.model_v = None
        self.scaler = None
        self.is_trained = False
        
    def extract_features(self, lat: float, lon: float, u: float, v: float, 
                        timestamp: datetime) -> np.array:
        """Extract features for ML model"""
        features = [
            lat, lon, u, v,
            timestamp.hour,
            timestamp.day,
            timestamp.month,
            np.sin(2 * np.pi * timestamp.hour / 24),  # Time of day cycle
            np.cos(2 * np.pi * timestamp.hour / 24),
            np.sin(2 * np.pi * timestamp.day / 365),   # Seasonal cycle
            np.cos(2 * np.pi * timestamp.day / 365),
            np.sqrt(u**2 + v**2),  # Current magnitude
            np.arctan2(v, u)       # Current direction
        ]
        return np.array(features).reshape(1, -1)
    
    def train_model(self, training_data: pd.DataFrame):
        """Train ML model on historical drift data"""
        if not ML_AVAILABLE:
            self.logger.warning("ML libraries not available - using basic physics model")
            return
        
        if training_data.empty:
            self.logger.warning("No training data available")
            return
        
        try:
            # Prepare features and targets
            features = []
            targets_u = []
            targets_v = []
            
            for _, row in training_data.iterrows():
                if pd.notna([row['lat'], row['lon'], row['u'], row['v'], 
                           row['predicted_lat'], row['predicted_lon']]).all():
                    
                    timestamp = pd.to_datetime(row['timestamp'])
                    feature_vec = self.extract_features(
                        row['lat'], row['lon'], row['u'], row['v'], timestamp
                    ).flatten()
                    
                    features.append(feature_vec)
                    targets_u.append(row['predicted_lat'] - row['lat'])
                    targets_v.append(row['predicted_lon'] - row['lon'])
            
            if len(features) < 10:
                self.logger.warning("Insufficient training data for ML model")
                return
            
            X = np.array(features)
            y_u = np.array(targets_u)
            y_v = np.array(targets_v)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.model_u = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_v = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.model_u.fit(X_scaled, y_u)
            self.model_v.fit(X_scaled, y_v)
            
            self.is_trained = True
            self.logger.info(f"ML model trained on {len(features)} samples")
            
            # Save models
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            joblib.dump(self.model_u, model_dir / "drift_model_u.pkl")
            joblib.dump(self.model_v, model_dir / "drift_model_v.pkl")
            joblib.dump(self.scaler, model_dir / "feature_scaler.pkl")
            
        except Exception as e:
            self.logger.error(f"Error training ML model: {e}")
    
    def load_model(self):
        """Load pre-trained model"""
        if not ML_AVAILABLE:
            return
        
        try:
            model_dir = Path("models")
            if (model_dir / "drift_model_u.pkl").exists():
                self.model_u = joblib.load(model_dir / "drift_model_u.pkl")
                self.model_v = joblib.load(model_dir / "drift_model_v.pkl")
                self.scaler = joblib.load(model_dir / "feature_scaler.pkl")
                self.is_trained = True
                self.logger.info("Pre-trained ML model loaded")
        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}")
    
    def predict_correction(self, lat: float, lon: float, u: float, v: float,
                          timestamp: datetime) -> Tuple[float, float]:
        """Predict ML correction to physics-based drift"""
        if not self.is_trained or not ML_AVAILABLE:
            return 0.0, 0.0
        
        try:
            features = self.extract_features(lat, lon, u, v, timestamp)
            features_scaled = self.scaler.transform(features)
            
            correction_u = self.model_u.predict(features_scaled)[0]
            correction_v = self.model_v.predict(features_scaled)[0]
            
            return correction_u, correction_v
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return 0.0, 0.0

class DriftEngine:
    """Enhanced drift calculation engine"""
    
    def __init__(self, data_manager: DataManager, logger: Logger):
        self.data_manager = data_manager
        self.logger = logger
        self.ml_model = MLEnhancedDriftModel(data_manager, logger)
        self.ml_model.load_model()
    
    def interpolate_currents(self, currents_df: pd.DataFrame) -> Dict:
        """Create interpolation functions for current data"""
        if currents_df.empty:
            return {}
        
        # Convert time to numeric
        currents_df['time_numeric'] = pd.to_datetime(currents_df['timestamp']).astype('int64') // 10**9
        
        # Group by time
        time_groups = currents_df.groupby('time_numeric')
        interpolators = {}
        
        for time_val, group in time_groups:
            if len(group) < 4:  # Need at least 4 points for interpolation
                continue
            
            try:
                # Create 2D interpolation functions
                lat_vals = group['lat'].values
                lon_vals = group['lon'].values
                u_vals = group['u'].values
                v_vals = group['v'].values
                
                # Remove any NaN values
                mask = ~(np.isnan(lat_vals) | np.isnan(lon_vals) | 
                        np.isnan(u_vals) | np.isnan(v_vals))
                
                if np.sum(mask) < 4:
                    continue
                
                lat_vals = lat_vals[mask]
                lon_vals = lon_vals[mask]
                u_vals = u_vals[mask]
                v_vals = v_vals[mask]
                
                # Create regular grid for interpolation
                lat_min, lat_max = lat_vals.min(), lat_vals.max()
                lon_min, lon_max = lon_vals.min(), lon_vals.max()
                
                if lat_max - lat_min < 0.01 or lon_max - lon_min < 0.01:
                    continue
                
                # Simple nearest neighbor approach if grid is too sparse
                from scipy.spatial import cKDTree
                points = np.column_stack((lat_vals, lon_vals))
                tree = cKDTree(points)
                
                interpolators[time_val] = {
                    'tree': tree,
                    'u_values': u_vals,
                    'v_values': v_vals
                }
                
            except Exception as e:
                self.logger.debug(f"Error creating interpolator for time {time_val}: {e}")
                continue
        
        return interpolators
    
    def get_current_at_point(self, lat: float, lon: float, timestamp: datetime,
                           interpolators: Dict) -> Tuple[float, float]:
        """Get interpolated current at specific point and time"""
        time_numeric = timestamp.timestamp()
        
        if not interpolators:
            return 0.0, 0.0
        
        # Find nearest time
        available_times = list(interpolators.keys())
        nearest_time = min(available_times, key=lambda t: abs(t - time_numeric))
        
        if abs(nearest_time - time_numeric) > 3600:  # More than 1 hour difference
            self.logger.warning(f"Large time gap in current data: {abs(nearest_time - time_numeric)/3600:.1f} hours")
        
        interp_data = interpolators[nearest_time]
        
        try:
            # Query nearest neighbors
            distances, indices = interp_data['tree'].query([lat, lon], k=min(4, len(interp_data['u_values'])))
            
            if np.isscalar(indices):
                indices = [indices]
                distances = [distances]
            
            # Inverse distance weighting
            weights = 1.0 / (distances + 1e-10)  # Small constant to avoid division by zero
            weights /= weights.sum()
            
            u_interp = np.sum(weights * interp_data['u_values'][indices])
            v_interp = np.sum(weights * interp_data['v_values'][indices])
            
            return float(u_interp), float(v_interp)
            
        except Exception as e:
            self.logger.debug(f"Error in current interpolation: {e}")
            return 0.0, 0.0
    
    def step_particle(self, lat: float, lon: float, u: float, v: float,
                     dt_seconds: float, windage: float = 0.0, 
                     stokes: float = 0.0, timestamp: datetime = None) -> Tuple[float, float]:
        """Advance particle one time step"""
        
        # Apply windage and Stokes drift
        u_total = u * (1.0 + stokes) + u * windage
        v_total = v * (1.0 + stokes) + v * windage
        
        # Apply ML correction if available
        if timestamp and self.ml_model.is_trained:
            correction_lat, correction_lon = self.ml_model.predict_correction(
                lat, lon, u, v, timestamp
            )
            # Scale corrections by time step
            correction_lat *= dt_seconds / 3600.0  # Assuming corrections are per hour
            correction_lon *= dt_seconds / 3600.0
        else:
            correction_lat = correction_lon = 0.0
        
        # Earth radius in meters
        R_earth = 6371000.0
        
        # Convert velocity to displacement
        dlat = (v_total * dt_seconds / R_earth) * (180.0 / np.pi) + correction_lat
        dlon = (u_total * dt_seconds / (R_earth * np.cos(np.radians(lat)))) * (180.0 / np.pi) + correction_lon
        
        new_lat = lat + dlat
        new_lon = lon + dlon
        
        # Keep longitude in valid range
        new_lon = ((new_lon + 180) % 360) - 180
        
        # Clamp latitude
        new_lat = max(-90, min(90, new_lat))
        
        return new_lat, new_lon
    
    def run_drift_simulation(self, seeds: pd.DataFrame, currents: pd.DataFrame,
                           dt_minutes: int = 10, duration_hours: int = 24,
                           windage: float = 0.03, stokes: float = 0.01,
                           reverse: bool = False) -> Dict[int, List[Tuple[float, float]]]:
        """Run drift simulation for multiple particles"""
        
        # Prepare current interpolators
        interpolators = self.interpolate_currents(currents)
        
        if not interpolators:
            self.logger.warning("No current data available - using zero currents")
        
        dt_seconds = dt_minutes * 60
        num_steps = int((duration_hours * 3600) / dt_seconds)
        
        if reverse:
            dt_seconds = -dt_seconds
        
        tracks = {}
        
        for idx, seed in seeds.iterrows():
            track = [(float(seed['lon']), float(seed['lat']))]
            
            lat, lon = float(seed['lat']), float(seed['lon'])
            start_time = pd.to_datetime(seed['time_iso'])
            
            for step in range(num_steps):
                current_time = start_time + timedelta(seconds=step * abs(dt_seconds))
                
                # Get current at this location and time
                u, v = self.get_current_at_point(lat, lon, current_time, interpolators)
                
                if reverse:
                    u, v = -u, -v
                
                # Step particle
                lat, lon = self.step_particle(
                    lat, lon, u, v, abs(dt_seconds), windage, stokes, current_time
                )
                
                track.append((float(lon), float(lat)))
                
                # Store for ML training (if not reverse simulation)
                if not reverse and step > 0:
                    # This would be used for training - comparing predicted vs actual
                    pass
            
            tracks[idx] = track
        
        self.logger.info(f"Completed drift simulation for {len(tracks)} particles")
        return tracks

class ParticleSeeder:
    """Generate particle seeds for drift simulation"""
    
    @staticmethod
    def generate_circular_seeds(center_lat: float, center_lon: float,
                              radius_nm: float, start_time: str, end_time: str,
                              rate_per_hour: int = 60) -> pd.DataFrame:
        """Generate seeds in circular pattern"""
        
        # Parse times
        start_dt = pd.to_datetime(start_time.replace('Z', '+00:00'))
        end_dt = pd.to_datetime(end_time.replace('Z', '+00:00'))
        
        # Generate time series
        time_interval = timedelta(minutes=60 / rate_per_hour)
        times = []
        current_time = start_dt
        
        while current_time <= end_dt:
            times.append(current_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
            current_time += time_interval
        
        # Convert radius from nautical miles to degrees
        radius_deg = radius_nm / 60.0
        
        # Generate random positions within circle
        np.random.seed(42)  # For reproducibility
        n_seeds = len(times)
        
        # Uniform distribution within circle
        r = radius_deg * np.sqrt(np.random.uniform(0, 1, n_seeds))
        theta = np.random.uniform(0, 2*np.pi, n_seeds)
        
        # Convert to lat/lon
        lats = center_lat + r * np.cos(theta)
        lons = center_lon + r * np.sin(theta) / np.cos(np.radians(center_lat))
        
        return pd.DataFrame({
            'time_iso': times,
            'lat': lats,
            'lon': lons
        })
    
    @staticmethod
    def generate_line_seeds(start_lat: float, start_lon: float,
                          end_lat: float, end_lon: float,
                          num_seeds: int, start_time: str) -> pd.DataFrame:
        """Generate seeds along a line (e.g., search pattern)"""
        
        lats = np.linspace(start_lat, end_lat, num_seeds)
        lons = np.linspace(start_lon, end_lon, num_seeds)
        times = [start_time] * num_seeds
        
        return pd.DataFrame({
            'time_iso': times,
            'lat': lats,
            'lon': lons
        })

class CESAROPSApp:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CESAROPS - SAR Drift Modeling")
        self.root.geometry("1000x800")
        
        # Initialize components
        self.logger = Logger()
        self.data_manager = DataManager()
        self.currents_fetcher = CurrentsFetcher(self.data_manager, self.logger)
        self.drift_engine = DriftEngine(self.data_manager, self.logger)
        
        # Load configuration
        self.load_config()
        
        # Create GUI
        self.create_widgets()
        
        # Message queue for thread communication
        self.message_queue = queue.Queue()
        self.root.after(100, self.process_messages)
    
    def load_config(self):
        """Load configuration from YAML file"""
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'erddap': {
                    'lmhofs': 'https://coastwatch.glerl.noaa.gov/erddap',
                    'rtofs': 'https://coastwatch.pfeg.noaa.gov/erddap',
                    'hycom': 'https://tds.hycom.org/erddap'
                },
                'drift_defaults': {
                    'dt_minutes': 10,
                    'duration_hours': 24,
                    'windage': 0.03,
                    'stokes': 0.01
                },
                'seeding': {
                    'default_radius_nm': 2.0,
                    'default_rate': 60
                }
            }
    
    def create_widgets(self):
        """Create main GUI interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_data_tab()
        self.create_seeding_tab()
        self.create_simulation_tab()
        self.create_results_tab()
        self.create_settings_tab()
        
        # Status bar and log
        self.create_status_bar()
        self.create_log_area()
    
    def create_data_tab(self):
        """Data fetching and management tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Data Sources")
        
        # Current data section
        currents_frame = ttk.LabelFrame(frame, text="Ocean Current Data", padding=10)
        currents_frame.pack(fill='x', padx=5, pady=5)
        
        # Data source selection
        ttk.Label(currents_frame, text="Data Source:").grid(row=0, column=0, sticky='w')
        self.source_var = tk.StringVar(value="LMHOFS")
        source_combo = ttk.Combobox(currents_frame, textvariable=self.source_var, 
                                   values=list(self.currents_fetcher.KNOWN_SOURCES.keys()))
        source_combo.grid(row=0, column=1, padx=5, sticky='ew')
        
        # Time range
        ttk.Label(currents_frame, text="Start Time (UTC):").grid(row=1, column=0, sticky='w')
        self.start_time_var = tk.StringVar(value=(datetime.now(timezone.utc) - timedelta(hours=6)).strftime('%Y-%m-%dT%H:00:00Z'))
        ttk.Entry(currents_frame, textvariable=self.start_time_var, width=20).grid(row=1, column=1, padx=5, sticky='ew')
        
        ttk.Label(currents_frame, text="End Time (UTC):").grid(row=2, column=0, sticky='w')
        self.end_time_var = tk.StringVar(value=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:00:00Z'))
        ttk.Entry(currents_frame, textvariable=self.end_time_var, width=20).grid(row=2, column=1, padx=5, sticky='ew')
        
        # Bounding box
        ttk.Label(currents_frame, text="Bounding Box (W,E,S,N):").grid(row=3, column=0, sticky='w')
        self.bbox_var = tk.StringVar(value="-87.0,-85.0,41.5,43.5")
        ttk.Entry(currents_frame, textvariable=self.bbox_var, width=30).grid(row=3, column=1, padx=5, sticky='ew')
        
        # Fetch button
        fetch_btn = ttk.Button(currents_frame, text="Fetch Current Data", 
                              command=self.fetch_currents_async)
        fetch_btn.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Data status
        self.data_status_var = tk.StringVar(value="No current data loaded")
        ttk.Label(currents_frame, textvariable=self.data_status_var).grid(row=5, column=0, columnspan=2)
        
        # Cached data section
        cache_frame = ttk.LabelFrame(frame, text="Cached Data", padding=10)
        cache_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(cache_frame, text="View Cached Data", 
                  command=self.view_cached_data).pack(side='left', padx=5)
        ttk.Button(cache_frame, text="Clear Cache", 
                  command=self.clear_cache).pack(side='left', padx=5)
        ttk.Button(cache_frame, text="Export Data", 
                  command=self.export_data).pack(side='left', padx=5)
        
        currents_frame.columnconfigure(1, weight=1)
    
    def create_seeding_tab(self):
        """Particle seeding configuration tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Seed Particles")
        
        # Circular seeding
        circular_frame = ttk.LabelFrame(frame, text="Circular Seeding Pattern", padding=10)
        circular_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(circular_frame, text="Center Latitude:").grid(row=0, column=0, sticky='w')
        self.center_lat_var = tk.StringVar(value="42.435833")
        ttk.Entry(circular_frame, textvariable=self.center_lat_var, width=15).grid(row=0, column=1, padx=5, sticky='ew')
        
        ttk.Label(circular_frame, text="Center Longitude:").grid(row=0, column=2, sticky='w')
        self.center_lon_var = tk.StringVar(value="-86.265556")
        ttk.Entry(circular_frame, textvariable=self.center_lon_var, width=15).grid(row=0, column=3, padx=5, sticky='ew')
        
        ttk.Label(circular_frame, text="Radius (nautical miles):").grid(row=1, column=0, sticky='w')
        self.radius_var = tk.StringVar(value="2.0")
        ttk.Entry(circular_frame, textvariable=self.radius_var, width=10).grid(row=1, column=1, padx=5, sticky='ew')
        
        ttk.Label(circular_frame, text="Seeds per hour:").grid(row=1, column=2, sticky='w')
        self.seed_rate_var = tk.StringVar(value="60")
        ttk.Entry(circular_frame, textvariable=self.seed_rate_var, width=10).grid(row=1, column=3, padx=5, sticky='ew')
        
        ttk.Label(circular_frame, text="Start Time:").grid(row=2, column=0, sticky='w')
        self.seed_start_var = tk.StringVar(value=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:00:00Z'))
        ttk.Entry(circular_frame, textvariable=self.seed_start_var, width=20).grid(row=2, column=1, columnspan=2, padx=5, sticky='ew')
        
        ttk.Label(circular_frame, text="End Time:").grid(row=3, column=0, sticky='w')
        self.seed_end_var = tk.StringVar(value=(datetime.now(timezone.utc) + timedelta(hours=1)).strftime('%Y-%m-%dT%H:00:00Z'))
        ttk.Entry(circular_frame, textvariable=self.seed_end_var, width=20).grid(row=3, column=1, columnspan=2, padx=5, sticky='ew')
        
        ttk.Button(circular_frame, text="Generate Seeds", 
                  command=self.generate_seeds).grid(row=4, column=0, columnspan=4, pady=10)
        
        # Line seeding
        line_frame = ttk.LabelFrame(frame, text="Line Seeding Pattern", padding=10)
        line_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(line_frame, text="Start Lat,Lon:").grid(row=0, column=0, sticky='w')
        self.line_start_var = tk.StringVar(value="42.4,-86.3")
        ttk.Entry(line_frame, textvariable=self.line_start_var, width=20).grid(row=0, column=1, padx=5, sticky='ew')
        
        ttk.Label(line_frame, text="End Lat,Lon:").grid(row=0, column=2, sticky='w')
        self.line_end_var = tk.StringVar(value="42.5,-86.2")
        ttk.Entry(line_frame, textvariable=self.line_end_var, width=20).grid(row=0, column=3, padx=5, sticky='ew')
        
        ttk.Label(line_frame, text="Number of seeds:").grid(row=1, column=0, sticky='w')
        self.line_count_var = tk.StringVar(value="10")
        ttk.Entry(line_frame, textvariable=self.line_count_var, width=10).grid(row=1, column=1, padx=5, sticky='ew')
        
        ttk.Button(line_frame, text="Generate Line Seeds", 
                  command=self.generate_line_seeds).grid(row=2, column=0, columnspan=4, pady=10)
        
        # Seed status
        self.seed_status_var = tk.StringVar(value="No seeds generated")
        ttk.Label(frame, textvariable=self.seed_status_var).pack(pady=5)
        
        # Configure column weights
        circular_frame.columnconfigure((1,3), weight=1)
        line_frame.columnconfigure((1,3), weight=1)
    
    def create_simulation_tab(self):
        """Drift simulation configuration tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Run Simulation")
        
        # Simulation parameters
        params_frame = ttk.LabelFrame(frame, text="Simulation Parameters", padding=10)
        params_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(params_frame, text="Time step (minutes):").grid(row=0, column=0, sticky='w')
        self.dt_var = tk.StringVar(value=str(self.config['drift_defaults']['dt_minutes']))
        ttk.Entry(params_frame, textvariable=self.dt_var, width=10).grid(row=0, column=1, padx=5, sticky='ew')
        
        ttk.Label(params_frame, text="Duration (hours):").grid(row=0, column=2, sticky='w')
        self.duration_var = tk.StringVar(value=str(self.config['drift_defaults']['duration_hours']))
        ttk.Entry(params_frame, textvariable=self.duration_var, width=10).grid(row=0, column=3, padx=5, sticky='ew')
        
        ttk.Label(params_frame, text="Windage factor:").grid(row=1, column=0, sticky='w')
        self.windage_var = tk.StringVar(value=str(self.config['drift_defaults']['windage']))
        ttk.Entry(params_frame, textvariable=self.windage_var, width=10).grid(row=1, column=1, padx=5, sticky='ew')
        
        ttk.Label(params_frame, text="Stokes drift factor:").grid(row=1, column=2, sticky='w')
        self.stokes_var = tk.StringVar(value=str(self.config['drift_defaults']['stokes']))
        ttk.Entry(params_frame, textvariable=self.stokes_var, width=10).grid(row=1, column=3, padx=5, sticky='ew')
        
        # ML Enhancement
        ml_frame = ttk.LabelFrame(frame, text="Machine Learning Enhancement", padding=10)
        ml_frame.pack(fill='x', padx=5, pady=5)
        
        self.use_ml_var = tk.BooleanVar(value=ML_AVAILABLE)
        ttk.Checkbutton(ml_frame, text="Use ML-enhanced predictions", 
                       variable=self.use_ml_var, 
                       state='normal' if ML_AVAILABLE else 'disabled').pack(anchor='w')
        
        if not ML_AVAILABLE:
            ttk.Label(ml_frame, text="ML libraries not installed - install scikit-learn and joblib for ML features", 
                     foreground='orange').pack(anchor='w')
        
        ttk.Button(ml_frame, text="Train ML Model", 
                  command=self.train_ml_model,
                  state='normal' if ML_AVAILABLE else 'disabled').pack(side='left', padx=5)
        
        ttk.Button(ml_frame, text="View ML Performance", 
                  command=self.view_ml_performance,
                  state='normal' if ML_AVAILABLE else 'disabled').pack(side='left', padx=5)
        
        # Simulation controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', padx=5, pady=10)
        
        ttk.Button(control_frame, text="Run Forward Drift", 
                  command=self.run_forward_simulation, 
                  style='Accent.TButton').pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="Run Backtrack", 
                  command=self.run_backtrack_simulation).pack(side='left', padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        # Simulation status
        self.sim_status_var = tk.StringVar(value="Ready to run simulation")
        ttk.Label(frame, textvariable=self.sim_status_var).pack(pady=5)
        
        params_frame.columnconfigure((1,3), weight=1)
    
    def create_results_tab(self):
        """Results viewing and export tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Results")
        
        # Results display
        display_frame = ttk.LabelFrame(frame, text="Simulation Results", padding=10)
        display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results summary
        self.results_summary_var = tk.StringVar(value="No simulation results available")
        ttk.Label(display_frame, textvariable=self.results_summary_var).pack(anchor='w', pady=5)
        
        # Export controls
        export_frame = ttk.Frame(display_frame)
        export_frame.pack(fill='x', pady=5)
        
        ttk.Button(export_frame, text="Export to CSV", 
                  command=self.export_tracks_csv).pack(side='left', padx=5)
        
        if KML_AVAILABLE:
            ttk.Button(export_frame, text="Export to KML", 
                      command=self.export_tracks_kml).pack(side='left', padx=5)
        
        ttk.Button(export_frame, text="Generate Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        
        # Visualization placeholder
        viz_frame = ttk.LabelFrame(display_frame, text="Track Visualization", padding=10)
        viz_frame.pack(fill='both', expand=True, pady=5)
        
        ttk.Label(viz_frame, text="Track visualization will appear here\n(Install matplotlib for enhanced plotting)", 
                 justify='center').pack(expand=True)
    
    def create_settings_tab(self):
        """Configuration and settings tab"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Settings")
        
        # Data sources configuration
        sources_frame = ttk.LabelFrame(frame, text="Data Sources", padding=10)
        sources_frame.pack(fill='x', padx=5, pady=5)
        
        for i, (name, config) in enumerate(self.config['erddap'].items()):
            ttk.Label(sources_frame, text=f"{name.upper()}:").grid(row=i, column=0, sticky='w')
            var = tk.StringVar(value=config)
            setattr(self, f"{name}_url_var", var)
            ttk.Entry(sources_frame, textvariable=var, width=50).grid(row=i, column=1, padx=5, sticky='ew')
        
        sources_frame.columnconfigure(1, weight=1)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(frame, text="Performance Settings", padding=10)
        perf_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(perf_frame, text="Cache retention (hours):").grid(row=0, column=0, sticky='w')
        self.cache_hours_var = tk.StringVar(value="24")
        ttk.Entry(perf_frame, textvariable=self.cache_hours_var, width=10).grid(row=0, column=1, padx=5, sticky='ew')
        
        ttk.Label(perf_frame, text="HTTP timeout (seconds):").grid(row=1, column=0, sticky='w')
        self.timeout_var = tk.StringVar(value="30")
        ttk.Entry(perf_frame, textvariable=self.timeout_var, width=10).grid(row=1, column=1, padx=5, sticky='ew')
        
        # Save settings
        ttk.Button(frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        
        perf_frame.columnconfigure(1, weight=1)
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.status_frame, textvariable=self.status_var, relief='sunken').pack(fill='x', padx=2, pady=2)
    
    def create_log_area(self):
        """Create expandable log area"""
        log_frame = ttk.LabelFrame(self.root, text="Activity Log", padding=5)
        log_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, state='disabled')
        self.log_text.pack(fill='both', expand=True)
        
        # Add custom log handler
        self.log_handler = GUILogHandler(self.log_text)
        logging.getLogger().addHandler(self.log_handler)
    
    def process_messages(self):
        """Process messages from background threads"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                message_type = message.get('type')
                
                if message_type == 'status':
                    self.status_var.set(message['text'])
                elif message_type == 'progress':
                    self.progress_var.set(message['value'])
                elif message_type == 'data_loaded':
                    self.data_status_var.set(message['text'])
                elif message_type == 'seeds_generated':
                    self.seed_status_var.set(message['text'])
                elif message_type == 'simulation_complete':
                    self.results_summary_var.set(message['text'])
                    self.current_tracks = message.get('tracks', {})
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_messages)
    
    # Event handlers
    def fetch_currents_async(self):
        """Fetch current data in background thread"""
        def fetch_worker():
            try:
                self.message_queue.put({'type': 'status', 'text': 'Fetching current data...'})
                
                source = self.source_var.get()
                start_time = self.start_time_var.get()
                end_time = self.end_time_var.get()
                bbox_str = self.bbox_var.get()
                
                # Parse bounding box
                bbox_parts = [float(x.strip()) for x in bbox_str.split(',')]
                bbox = tuple(bbox_parts)
                
                # Fetch data
                self.current_data = self.currents_fetcher.fetch_currents(
                    source, bbox, start_time, end_time
                )
                
                self.message_queue.put({
                    'type': 'data_loaded',
                    'text': f"Loaded {len(self.current_data)} current records from {source}"
                })
                self.message_queue.put({'type': 'status', 'text': 'Current data fetch complete'})
                
            except Exception as e:
                self.logger.error(f"Error fetching currents: {e}")
                self.message_queue.put({'type': 'status', 'text': f'Error: {str(e)}'})
        
        threading.Thread(target=fetch_worker, daemon=True).start()
    
    def generate_seeds(self):
        """Generate circular seed pattern"""
        try:
            center_lat = float(self.center_lat_var.get())
            center_lon = float(self.center_lon_var.get())
            radius = float(self.radius_var.get())
            rate = int(self.seed_rate_var.get())
            start_time = self.seed_start_var.get()
            end_time = self.seed_end_var.get()
            
            self.seeds = ParticleSeeder.generate_circular_seeds(
                center_lat, center_lon, radius, start_time, end_time, rate
            )
            
            self.seed_status_var.set(f"Generated {len(self.seeds)} seeds in circular pattern")
            self.logger.info(f"Generated {len(self.seeds)} seeds")
            
        except Exception as e:
            self.logger.error(f"Error generating seeds: {e}")
            messagebox.showerror("Error", f"Failed to generate seeds: {str(e)}")
    
    def generate_line_seeds(self):
        """Generate line seed pattern"""
        try:
            start_parts = [float(x.strip()) for x in self.line_start_var.get().split(',')]
            end_parts = [float(x.strip()) for x in self.line_end_var.get().split(',')]
            count = int(self.line_count_var.get())
            start_time = self.seed_start_var.get()
            
            self.seeds = ParticleSeeder.generate_line_seeds(
                start_parts[0], start_parts[1], end_parts[0], end_parts[1], count, start_time
            )
            
            self.seed_status_var.set(f"Generated {len(self.seeds)} seeds in line pattern")
            self.logger.info(f"Generated {len(self.seeds)} seeds in line")
            
        except Exception as e:
            self.logger.error(f"Error generating line seeds: {e}")
            messagebox.showerror("Error", f"Failed to generate line seeds: {str(e)}")
    
    def run_forward_simulation(self):
        """Run forward drift simulation"""
        self._run_simulation(reverse=False)
    
    def run_backtrack_simulation(self):
        """Run backtrack simulation"""
        self._run_simulation(reverse=True)
    
    def _run_simulation(self, reverse=False):
        """Run drift simulation in background thread"""
        if not hasattr(self, 'seeds') or not hasattr(self, 'current_data'):
            messagebox.showerror("Error", "Please generate seeds and fetch current data first")
            return
        
        def sim_worker():
            try:
                sim_type = "Backtrack" if reverse else "Forward drift"
                self.message_queue.put({'type': 'status', 'text': f'Running {sim_type.lower()} simulation...'})
                
                dt_minutes = int(self.dt_var.get())
                duration_hours = int(self.duration_var.get())
                windage = float(self.windage_var.get())
                stokes = float(self.stokes_var.get())
                
                # Update progress
                for i in range(0, 101, 10):
                    self.message_queue.put({'type': 'progress', 'value': i})
                    threading.Event().wait(0.1)  # Simulate progress
                
                tracks = self.drift_engine.run_drift_simulation(
                    self.seeds, self.current_data,
                    dt_minutes=dt_minutes,
                    duration_hours=duration_hours,
                    windage=windage,
                    stokes=stokes,
                    reverse=reverse
                )
                
                self.message_queue.put({'type': 'progress', 'value': 100})
                self.message_queue.put({
                    'type': 'simulation_complete',
                    'text': f"{sim_type} complete: {len(tracks)} particle tracks generated",
                    'tracks': tracks
                })
                self.message_queue.put({'type': 'status', 'text': f'{sim_type} simulation complete'})
                
            except Exception as e:
                self.logger.error(f"Error in simulation: {e}")
                self.message_queue.put({'type': 'status', 'text': f'Simulation error: {str(e)}'})
        
        threading.Thread(target=sim_worker, daemon=True).start()
    
    def train_ml_model(self):
        """Train ML model on historical data"""
        if not ML_AVAILABLE:
            messagebox.showwarning("ML Not Available", "Machine learning libraries not installed")
            return
        
        def train_worker():
            try:
                self.message_queue.put({'type': 'status', 'text': 'Training ML model...'})
                
                # Get training data from database
                conn = sqlite3.connect(self.data_manager.db_path)
                training_df = pd.read_sql_query("SELECT * FROM drift_tracks", conn)
                conn.close()
                
                if training_df.empty:
                    self.message_queue.put({'type': 'status', 'text': 'No training data available'})
                    return
                
                self.drift_engine.ml_model.train_model(training_df)
                self.message_queue.put({'type': 'status', 'text': 'ML model training complete'})
                
            except Exception as e:
                self.logger.error(f"Error training ML model: {e}")
                self.message_queue.put({'type': 'status', 'text': f'ML training error: {str(e)}'})
        
        threading.Thread(target=train_worker, daemon=True).start()
    
    def view_cached_data(self):
        """View cached current data"""
        cached_data = self.data_manager.get_cached_data()
        if cached_data.empty:
            messagebox.showinfo("Cache", "No cached data available")
        else:
            # Show data summary
            summary = f"Cached data records: {len(cached_data)}\n"
            summary += f"Time range: {cached_data['timestamp'].min()} to {cached_data['timestamp'].max()}\n"
            summary += f"Sources: {', '.join(cached_data['source'].unique())}"
            messagebox.showinfo("Cached Data", summary)
    
    def clear_cache(self):
        """Clear cached data"""
        if messagebox.askyesno("Clear Cache", "Are you sure you want to clear all cached data?"):
            conn = sqlite3.connect(self.data_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM current_data")
            conn.commit()
            conn.close()
            self.logger.info("Cache cleared")
            messagebox.showinfo("Cache", "Cache cleared successfully")
    
    def export_tracks_csv(self):
        """Export drift tracks to CSV"""
        if not hasattr(self, 'current_tracks'):
            messagebox.showerror("Error", "No simulation results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Convert tracks to DataFrame
                rows = []
                for seed_id, track in self.current_tracks.items():
                    for step, (lon, lat) in enumerate(track):
                        rows.append({
                            'seed_id': seed_id,
                            'step': step,
                            'latitude': lat,
                            'longitude': lon
                        })
                
                df = pd.DataFrame(rows)
                df.to_csv(filename, index=False)
                
                self.logger.info(f"Tracks exported to {filename}")
                messagebox.showinfo("Export", f"Tracks exported to {filename}")
                
            except Exception as e:
                self.logger.error(f"Error exporting tracks: {e}")
                messagebox.showerror("Error", f"Failed to export tracks: {str(e)}")
    
    def export_tracks_kml(self):
        """Export drift tracks to KML"""
        if not KML_AVAILABLE:
            messagebox.showerror("Error", "KML support not available (install simplekml)")
            return
        
        if not hasattr(self, 'current_tracks'):
            messagebox.showerror("Error", "No simulation results to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                kml = simplekml.Kml()
                
                for seed_id, track in self.current_tracks.items():
                    line = kml.newlinestring(name=f"Track {seed_id}")
                line.coords = track
                line.style.linestyle.color = simplekml.Color.red
                line.style.linestyle.width = 2
                
                kml.save(filename)
                
                self.logger.info(f"Tracks exported to KML: {filename}")
                messagebox.showinfo("Export", f"Tracks exported to {filename}")
                
            except Exception as e:
                self.logger.error(f"Error exporting KML: {e}")
                messagebox.showerror("Error", f"Failed to export KML: {str(e)}")
    
    def generate_report(self):
        """Generate comprehensive drift analysis report"""
        if not hasattr(self, 'current_tracks'):
            messagebox.showerror("Error", "No simulation results available")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("CESAROPS Drift Analysis Report\n")
                    f.write("=" * 50 + "\n\n")
                    
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Number of particles: {len(self.current_tracks)}\n")
                    
                    if hasattr(self, 'seeds'):
                        f.write(f"Simulation duration: {self.duration_var.get()} hours\n")
                        f.write(f"Time step: {self.dt_var.get()} minutes\n")
                        f.write(f"Windage factor: {self.windage_var.get()}\n")
                        f.write(f"Stokes drift factor: {self.stokes_var.get()}\n")
                    
                    f.write("\nTrack Summary:\n")
                    f.write("-" * 20 + "\n")
                    
                    total_distance = 0
                    max_distance = 0
                    
                    for seed_id, track in self.current_tracks.items():
                        if len(track) > 1:
                            # Calculate track distance
                            distance = 0
                            for i in range(1, len(track)):
                                lon1, lat1 = track[i-1]
                                lon2, lat2 = track[i]
                                # Simple distance calculation (not great circle)
                                dx = (lon2 - lon1) * 111320 * np.cos(np.radians(lat1))
                                dy = (lat2 - lat1) * 110540
                                distance += np.sqrt(dx*dx + dy*dy)
                            
                            total_distance += distance
                            max_distance = max(max_distance, distance)
                            
                            f.write(f"Particle {seed_id}: {distance/1000:.2f} km\n")
                    
                    if len(self.current_tracks) > 0:
                        avg_distance = total_distance / len(self.current_tracks)
                        f.write(f"\nAverage drift distance: {avg_distance/1000:.2f} km\n")
                        f.write(f"Maximum drift distance: {max_distance/1000:.2f} km\n")
                    
                    f.write("\nData Sources Used:\n")
                    f.write("-" * 20 + "\n")
                    if hasattr(self, 'current_data'):
                        sources = self.current_data.get('source', ['Unknown']).unique() if 'source' in self.current_data.columns else ['Unknown']
                        for source in sources:
                            f.write(f"- {source}\n")
                
                self.logger.info(f"Report generated: {filename}")
                messagebox.showinfo("Report", f"Report generated: {filename}")
                
            except Exception as e:
                self.logger.error(f"Error generating report: {e}")
                messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
    
    def export_data(self):
        """Export current data to CSV"""
        if not hasattr(self, 'current_data'):
            messagebox.showerror("Error", "No current data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.current_data.to_csv(filename, index=False)
                self.logger.info(f"Current data exported to {filename}")
                messagebox.showinfo("Export", f"Data exported to {filename}")
            except Exception as e:
                self.logger.error(f"Error exporting data: {e}")
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")
    
    def view_ml_performance(self):
        """View ML model performance metrics"""
        if not ML_AVAILABLE:
            messagebox.showwarning("ML Not Available", "Machine learning libraries not installed")
            return
        
        try:
            conn = sqlite3.connect(self.data_manager.db_path)
            perf_df = pd.read_sql_query("SELECT * FROM model_performance ORDER BY test_date DESC", conn)
            conn.close()
            
            if perf_df.empty:
                messagebox.showinfo("ML Performance", "No model performance data available")
            else:
                # Show latest performance
                latest = perf_df.iloc[0]
                msg = f"Latest ML Model Performance:\n\n"
                msg += f"Model Version: {latest['model_version']}\n"
                msg += f"Accuracy Score: {latest['accuracy_score']:.4f}\n"
                msg += f"Mean Absolute Error: {latest['mae']:.4f}\n"
                msg += f"Root Mean Square Error: {latest['rmse']:.4f}\n"
                msg += f"Test Date: {latest['test_date']}\n"
                messagebox.showinfo("ML Performance", msg)
        
        except Exception as e:
            self.logger.error(f"Error viewing ML performance: {e}")
            messagebox.showerror("Error", f"Failed to view ML performance: {str(e)}")
    
    def save_settings(self):
        """Save configuration settings"""
        try:
            # Update config with current values
            for name in self.config['erddap'].keys():
                var = getattr(self, f"{name}_url_var")
                self.config['erddap'][name] = var.get()
            
            # Save to file
            with open('config.yaml', 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info("Settings saved")
            messagebox.showinfo("Settings", "Settings saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def run(self):
        """Start the GUI application"""
        self.logger.info("CESAROPS application started")
        self.root.mainloop()


class GUILogHandler(logging.Handler):
    """Custom log handler that writes to GUI text widget"""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.text_widget.config(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.config(state='disabled')
        except Exception:
            pass


def create_default_config():
    """Create default configuration file"""
    config = {
        'erddap': {
            'lmhofs': 'https://coastwatch.glerl.noaa.gov/erddap',
            'rtofs': 'https://coastwatch.pfeg.noaa.gov/erddap',
            'hycom': 'https://tds.hycom.org/erddap'
        },
        'drift_defaults': {
            'dt_minutes': 10,
            'duration_hours': 24,
            'windage': 0.03,
            'stokes': 0.01
        },
        'seeding': {
            'default_radius_nm': 2.0,
            'default_rate': 60
        }
    }
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config


def main():
    """Main entry point"""
    print("Starting CESAROPS - Civilian Emergency SAR Operations")
    print("=" * 60)
    
    # Create config if it doesn't exist
    if not Path('config.yaml').exists():
        print("Creating default configuration...")
        create_default_config()
    
    # Create directories
    Path('data').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    try:
        # Start GUI application
        app = CESAROPSApp()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()

                