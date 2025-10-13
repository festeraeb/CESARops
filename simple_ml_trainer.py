#!/usr/bin/env python3
"""
Simple ML Training for Drifter Data
===================================

Train machine learning models on real drifter trajectory data
for improved drift predictions in Great Lakes SAR operations.

Uses the collected drifter data to train:
1. Simple regression model for drift prediction
2. Random Forest for trajectory correction
3. Basic neural network (if TensorFlow available)

Author: GitHub Copilot
Date: January 7, 2025
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for optional ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using simple regression")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available, skipping neural network training")

class SimpleDrifterMLTrainer:
    """Simple ML trainer for drifter data"""
    
    def __init__(self, db_file='drift_objects.db', models_dir='models'):
        self.db_file = db_file
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.trained_models = {}
        self.training_metrics = {}
    
    def load_training_data(self) -> pd.DataFrame:
        """Load drifter trajectory data from database"""
        logger.info("📊 Loading training data from database...")
        
        try:
            conn = sqlite3.connect(self.db_file)
            
            # Load trajectory data
            query = '''
                SELECT source, drifter_id, timestamp, latitude, longitude,
                       velocity_u, velocity_v, sea_surface_temp
                FROM real_drifter_trajectories
                ORDER BY drifter_id, timestamp
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                logger.error("No training data found in database")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate derived features
            df = self._calculate_features(df)
            
            logger.info(f"✅ Loaded {len(df)} trajectory points from {df['drifter_id'].nunique()} drifters")
            return df
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return pd.DataFrame()
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for ML training"""
        logger.info("⚙️ Calculating ML features...")
        
        # Sort by drifter and time
        df = df.sort_values(['drifter_id', 'timestamp']).reset_index(drop=True)
        
        # Calculate time differences and distances
        df['time_delta'] = df.groupby('drifter_id')['timestamp'].diff().dt.total_seconds() / 3600.0  # hours
        df['lat_delta'] = df.groupby('drifter_id')['latitude'].diff()
        df['lon_delta'] = df.groupby('drifter_id')['longitude'].diff()
        
        # Calculate speeds from position changes
        df['speed_lat'] = df['lat_delta'] / df['time_delta'].replace(0, np.nan)
        df['speed_lon'] = df['lon_delta'] / df['time_delta'].replace(0, np.nan)
        
        # Calculate distance traveled
        df['distance_km'] = np.sqrt(df['lat_delta']**2 + df['lon_delta']**2) * 111.0  # Rough km conversion
        df['speed_kph'] = df['distance_km'] / df['time_delta'].replace(0, np.nan)
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        
        # Add position-based features
        df['lat_sin'] = np.sin(np.radians(df['latitude']))
        df['lat_cos'] = np.cos(np.radians(df['latitude']))
        df['lon_sin'] = np.sin(np.radians(df['longitude']))
        df['lon_cos'] = np.cos(np.radians(df['longitude']))
        
        # Drop rows with NaN values from calculations
        df = df.dropna()
        
        logger.info(f"✅ Features calculated, {len(df)} valid points remaining")
        return df
    
    def prepare_ml_datasets(self, df: pd.DataFrame):
        """Prepare datasets for ML training"""
        logger.info("🔧 Preparing ML datasets...")
        
        if df.empty:
            return None, None, None, None
        
        # Features for prediction
        feature_columns = [
            'latitude', 'longitude', 'hour', 'day_of_year', 'month',
            'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos'
        ]
        
        # Add velocity features if available
        if 'velocity_u' in df.columns and df['velocity_u'].notna().any():
            feature_columns.extend(['velocity_u', 'velocity_v'])
        
        # Add temperature if available
        if 'sea_surface_temp' in df.columns and df['sea_surface_temp'].notna().any():
            feature_columns.append('sea_surface_temp')
        
        # Target variables (next position)
        df['next_lat'] = df.groupby('drifter_id')['latitude'].shift(-1)
        df['next_lon'] = df.groupby('drifter_id')['longitude'].shift(-1)
        
        # Remove last point of each trajectory (no target)
        df = df.dropna(subset=['next_lat', 'next_lon'])
        
        if df.empty:
            logger.error("No valid training examples after preprocessing")
            return None, None, None, None
        
        # Prepare features and targets
        X = df[feature_columns].values
        y = df[['next_lat', 'next_lon']].values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"✅ Dataset prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
        logger.info(f"📋 Features: {feature_columns}")
        
        return X_train, X_test, y_train, y_test
    
    def train_simple_regression(self, X_train, y_train, X_test, y_test):
        """Train simple linear regression model"""
        logger.info("🧠 Training simple regression model...")
        
        try:
            # Simple linear regression using numpy
            # Add bias term
            X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
            X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
            
            # Solve normal equations: θ = (X^T X)^(-1) X^T y
            weights_lat = np.linalg.lstsq(X_train_bias, y_train[:, 0], rcond=None)[0]
            weights_lon = np.linalg.lstsq(X_train_bias, y_train[:, 1], rcond=None)[0]
            
            # Predictions
            y_pred_lat = X_test_bias @ weights_lat
            y_pred_lon = X_test_bias @ weights_lon
            y_pred = np.column_stack([y_pred_lat, y_pred_lon])
            
            # Calculate metrics
            mse_lat = np.mean((y_test[:, 0] - y_pred_lat) ** 2)
            mse_lon = np.mean((y_test[:, 1] - y_pred_lon) ** 2)
            mse_total = np.mean(np.sum((y_test - y_pred) ** 2, axis=1))
            
            # R-squared
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            # Save model
            model_data = {
                'weights_lat': weights_lat.tolist(),
                'weights_lon': weights_lon.tolist(),
                'model_type': 'simple_regression',
                'training_date': datetime.now().isoformat(),
                'training_samples': X_train.shape[0]
            }
            
            with open(self.models_dir / 'simple_regression_model.json', 'w') as f:
                json.dump(model_data, f, indent=2)
            
            self.trained_models['simple_regression'] = model_data
            self.training_metrics['simple_regression'] = {
                'mse_lat': mse_lat,
                'mse_lon': mse_lon,
                'mse_total': mse_total,
                'r2_score': r2,
                'training_samples': X_train.shape[0]
            }
            
            logger.info(f"✅ Simple regression trained:")
            logger.info(f"   MSE (lat): {mse_lat:.6f}")
            logger.info(f"   MSE (lon): {mse_lon:.6f}")
            logger.info(f"   R² score: {r2:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training simple regression: {e}")
            return False
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, skipping Random Forest")
            return False
        
        logger.info("🌲 Training Random Forest model...")
        
        try:
            # Separate models for lat and lon
            rf_lat = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_lon = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Train models
            rf_lat.fit(X_train, y_train[:, 0])
            rf_lon.fit(X_train, y_train[:, 1])
            
            # Predictions
            y_pred_lat = rf_lat.predict(X_test)
            y_pred_lon = rf_lon.predict(X_test)
            y_pred = np.column_stack([y_pred_lat, y_pred_lon])
            
            # Calculate metrics
            mse_lat = mean_squared_error(y_test[:, 0], y_pred_lat)
            mse_lon = mean_squared_error(y_test[:, 1], y_pred_lon)
            mse_total = np.mean(np.sum((y_test - y_pred) ** 2, axis=1))
            
            r2_lat = r2_score(y_test[:, 0], y_pred_lat)
            r2_lon = r2_score(y_test[:, 1], y_pred_lon)
            r2_avg = (r2_lat + r2_lon) / 2
            
            # Save models
            with open(self.models_dir / 'random_forest_lat.pkl', 'wb') as f:
                pickle.dump(rf_lat, f)
            
            with open(self.models_dir / 'random_forest_lon.pkl', 'wb') as f:
                pickle.dump(rf_lon, f)
            
            self.trained_models['random_forest'] = {
                'model_lat_file': 'random_forest_lat.pkl',
                'model_lon_file': 'random_forest_lon.pkl',
                'model_type': 'random_forest',
                'training_date': datetime.now().isoformat(),
                'training_samples': X_train.shape[0]
            }
            
            self.training_metrics['random_forest'] = {
                'mse_lat': mse_lat,
                'mse_lon': mse_lon,
                'mse_total': mse_total,
                'r2_lat': r2_lat,
                'r2_lon': r2_lon,
                'r2_avg': r2_avg,
                'training_samples': X_train.shape[0]
            }
            
            logger.info(f"✅ Random Forest trained:")
            logger.info(f"   MSE (lat): {mse_lat:.6f}")
            logger.info(f"   MSE (lon): {mse_lon:.6f}")
            logger.info(f"   R² (lat): {r2_lat:.4f}")
            logger.info(f"   R² (lon): {r2_lon:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            return False
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train simple neural network"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, skipping neural network")
            return False
        
        logger.info("🤖 Training neural network...")
        
        try:
            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Simple neural network
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(2)  # lat, lon output
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train with early stopping
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_test_scaled, verbose=0)
            
            mse_total = np.mean(np.sum((y_test - y_pred) ** 2, axis=1))
            mae_total = np.mean(np.sum(np.abs(y_test - y_pred), axis=1))
            
            # R-squared
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            # Save model and scaler
            model.save(self.models_dir / 'neural_network_model')
            with open(self.models_dir / 'neural_network_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            self.trained_models['neural_network'] = {
                'model_file': 'neural_network_model',
                'scaler_file': 'neural_network_scaler.pkl',
                'model_type': 'neural_network',
                'training_date': datetime.now().isoformat(),
                'training_samples': X_train.shape[0]
            }
            
            self.training_metrics['neural_network'] = {
                'mse_total': mse_total,
                'mae_total': mae_total,
                'r2_score': r2,
                'final_val_loss': min(history.history['val_loss']),
                'training_samples': X_train.shape[0]
            }
            
            logger.info(f"✅ Neural network trained:")
            logger.info(f"   MSE: {mse_total:.6f}")
            logger.info(f"   MAE: {mae_total:.6f}")
            logger.info(f"   R² score: {r2:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            return False
    
    def save_training_report(self):
        """Save training report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'models_trained': list(self.trained_models.keys()),
            'training_metrics': self.training_metrics,
            'model_files': self.trained_models
        }
        
        with open(self.models_dir / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save human-readable report
        with open('ml_training_report.txt', 'w') as f:
            f.write("CESAROPS ML Training Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Training Date: {report['training_date']}\n")
            f.write(f"Models Trained: {', '.join(report['models_trained'])}\n\n")
            
            for model_name, metrics in self.training_metrics.items():
                f.write(f"{model_name.upper()} Model Results:\n")
                f.write("-" * 20 + "\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {metric}: {value:.6f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
                f.write("\n")
        
        logger.info("📄 Training report saved to ml_training_report.txt")
    
    def run_complete_training(self):
        """Run complete ML training pipeline"""
        logger.info("🚀 Starting complete ML training pipeline...")
        start_time = datetime.now()
        
        # Load data
        df = self.load_training_data()
        if df.empty:
            logger.error("❌ No training data available")
            return False
        
        # Prepare datasets
        X_train, X_test, y_train, y_test = self.prepare_ml_datasets(df)
        if X_train is None:
            logger.error("❌ Failed to prepare training datasets")
            return False
        
        trained_count = 0
        
        # Train models
        if self.train_simple_regression(X_train, y_train, X_test, y_test):
            trained_count += 1
        
        if self.train_random_forest(X_train, y_train, X_test, y_test):
            trained_count += 1
        
        if self.train_neural_network(X_train, y_train, X_test, y_test):
            trained_count += 1
        
        # Save report
        if trained_count > 0:
            self.save_training_report()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"🎉 Training completed in {duration:.1f} seconds")
        logger.info(f"📊 Successfully trained {trained_count} models")
        
        return trained_count > 0

def main():
    """Main function"""
    print("CESAROPS Simple ML Training")
    print("=" * 30)
    
    trainer = SimpleDrifterMLTrainer()
    success = trainer.run_complete_training()
    
    if success:
        print("\n✅ ML training completed successfully!")
        print("📊 Model Performance Summary:")
        
        for model_name, metrics in trainer.training_metrics.items():
            print(f"\n{model_name.upper()}:")
            if 'r2_score' in metrics:
                print(f"  R² Score: {metrics['r2_score']:.4f}")
            if 'r2_avg' in metrics:
                print(f"  Avg R² Score: {metrics['r2_avg']:.4f}")
            if 'mse_total' in metrics:
                print(f"  Total MSE: {metrics['mse_total']:.6f}")
            print(f"  Training Samples: {metrics['training_samples']}")
        
        print(f"\n📁 Models saved to: models/")
        print(f"📄 Report saved to: ml_training_report.txt")
    else:
        print("\n❌ ML training failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())