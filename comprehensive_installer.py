#!/usr/bin/env python3
"""
Comprehensive Installation and Setup System for CESAROPS
========================================================

Automated installation system that:
- Detects and creates conda environments
- Installs all required dependencies
- Validates installations with diagnostic tests
- Sets up database and configuration
- Provides troubleshooting guidance

Based on research findings and operational requirements for
state-of-the-art SAR drift prediction system.

Author: GitHub Copilot
Date: January 7, 2025
License: MIT
"""

import os
import sys
import subprocess
import platform
import json
import sqlite3
import time
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cesarops_installation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CESAROPSInstaller:
    """Complete installation and setup system for CESAROPS"""
    
    def __init__(self):
        """Initialize installer with system detection"""
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.base_dir = Path(__file__).parent
        self.conda_env_name = "cesarops"
        
        # Installation status tracking
        self.installation_status = {
            'conda_available': False,
            'environment_created': False,
            'dependencies_installed': False,
            'database_initialized': False,
            'tests_passed': False,
            'ml_models_ready': False
        }
        
        # Required packages with versions
        self.required_packages = {
            'core': [
                'numpy>=1.21.0',
                'scipy>=1.7.0',
                'pandas>=1.3.0',
                'matplotlib>=3.4.0',
                'requests>=2.25.0',
                'netcdf4>=1.5.0',
                'xarray>=0.19.0',
                'pyyaml>=5.4.0'
            ],
            'ml': [
                'tensorflow>=2.8.0',
                'scikit-learn>=1.0.0',
                'joblib>=1.0.0',
                'sentence-transformers>=2.2.0',
                'transformers>=4.20.0',
                'torch>=1.12.0'
            ],
            'geo': [
                'cartopy>=0.20.0',
                'geopandas>=0.10.0',
                'shapely>=1.8.0',
                'pyproj>=3.2.0',
                'folium>=0.12.0'
            ],
            'optional': [
                'jupyter>=1.0.0',
                'ipykernel>=6.0.0',
                'plotly>=5.0.0',
                'dash>=2.0.0',
                'opendrift>=1.8.0'
            ]
        }
        
        logger.info(f"Initialized CESAROPS installer on {self.system}")
    
    def check_conda_installation(self) -> bool:
        """Check if conda is available and functional"""
        try:
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                conda_version = result.stdout.strip()
                logger.info(f"Found conda: {conda_version}")
                self.installation_status['conda_available'] = True
                return True
            else:
                logger.error("Conda command failed")
                return False
        except FileNotFoundError:
            logger.error("Conda not found in PATH")
            return False
    
    def install_conda(self) -> bool:
        """Install miniconda if not available"""
        logger.info("Installing Miniconda...")
        
        # Determine installer URL based on system
        if self.system == 'windows':
            installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
            installer_file = "Miniconda3-latest-Windows-x86_64.exe"
        elif self.system == 'darwin':  # macOS
            installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
            installer_file = "Miniconda3-latest-MacOSX-x86_64.sh"
        else:  # Linux
            installer_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            installer_file = "Miniconda3-latest-Linux-x86_64.sh"
        
        try:
            # Download installer
            logger.info(f"Downloading {installer_file}...")
            urllib.request.urlretrieve(installer_url, installer_file)
            
            # Run installer
            if self.system == 'windows':
                logger.info("Please run the downloaded installer manually and restart this script")
                return False
            else:
                subprocess.run(['bash', installer_file, '-b', '-p', 
                              str(Path.home() / 'miniconda3')], check=True)
                
                # Add to PATH
                if self.system == 'darwin':
                    shell_rc = Path.home() / '.zshrc'
                else:
                    shell_rc = Path.home() / '.bashrc'
                
                with open(shell_rc, 'a') as f:
                    f.write('\n# Added by CESAROPS installer\n')
                    f.write('export PATH="$HOME/miniconda3/bin:$PATH"\n')
                
                logger.info("Miniconda installed. Please restart your terminal and run this script again.")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install conda: {e}")
            return False
        finally:
            # Clean up installer
            if os.path.exists(installer_file):
                os.remove(installer_file)
    
    def create_conda_environment(self) -> bool:
        """Create conda environment for CESAROPS"""
        try:
            # Check if environment already exists
            result = subprocess.run(['conda', 'env', 'list'], 
                                  capture_output=True, text=True)
            if self.conda_env_name in result.stdout:
                logger.info(f"Environment '{self.conda_env_name}' already exists")
                self.installation_status['environment_created'] = True
                return True
            
            # Create new environment
            logger.info(f"Creating conda environment '{self.conda_env_name}'...")
            subprocess.run([
                'conda', 'create', '-n', self.conda_env_name, 
                f'python={self.python_version}', '-y'
            ], check=True)
            
            self.installation_status['environment_created'] = True
            logger.info(f"Successfully created environment '{self.conda_env_name}'")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create conda environment: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install all required dependencies in conda environment"""
        try:
            # Get conda prefix
            result = subprocess.run([
                'conda', 'env', 'list', '--json'
            ], capture_output=True, text=True, check=True)
            
            env_list = json.loads(result.stdout)
            env_path = None
            for env in env_list['envs']:
                if self.conda_env_name in env:
                    env_path = env
                    break
            
            if not env_path:
                logger.error(f"Environment '{self.conda_env_name}' not found")
                return False
            
            # Install packages by category
            success = True
            
            # Core packages
            logger.info("Installing core packages...")
            success &= self._install_package_group('core', env_path)
            
            # ML packages
            logger.info("Installing ML packages...")
            success &= self._install_package_group('ml', env_path)
            
            # Geospatial packages
            logger.info("Installing geospatial packages...")
            success &= self._install_package_group('geo', env_path)
            
            # Optional packages (continue even if some fail)
            logger.info("Installing optional packages...")
            self._install_package_group('optional', env_path, required=False)
            
            if success:
                self.installation_status['dependencies_installed'] = True
                logger.info("All required dependencies installed successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def _install_package_group(self, group: str, env_path: str, required: bool = True) -> bool:
        """Install a group of packages"""
        packages = self.required_packages.get(group, [])
        
        for package in packages:
            try:
                logger.info(f"Installing {package}...")
                
                # Try conda first, then pip
                try:
                    subprocess.run([
                        'conda', 'install', '-n', self.conda_env_name, 
                        package, '-y'
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    # Fall back to pip
                    pip_path = os.path.join(env_path, 'Scripts' if self.system == 'windows' else 'bin', 'pip')
                    subprocess.run([
                        pip_path, 'install', package
                    ], check=True, capture_output=True)
                
                logger.info(f"Successfully installed {package}")
                
            except subprocess.CalledProcessError as e:
                if required:
                    logger.error(f"Failed to install required package {package}: {e}")
                    return False
                else:
                    logger.warning(f"Failed to install optional package {package}: {e}")
        
        return True
    
    def initialize_database(self) -> bool:
        """Initialize SQLite database with required tables"""
        try:
            db_path = self.base_dir / 'drift_objects.db'
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create tables for original CESAROPS
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_objects (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    object_type TEXT,
                    mass REAL,
                    area REAL,
                    windage REAL,
                    stokes_drift REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create tables for multi-modal system
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS multimodal_objects (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    mass REAL,
                    area_above_water REAL,
                    area_below_water REAL,
                    drag_coeff_air REAL,
                    drag_coeff_water REAL,
                    lift_coeff_air REAL,
                    lift_coeff_water REAL,
                    description TEXT,
                    object_type TEXT,
                    text_embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trajectory_predictions (
                    id INTEGER PRIMARY KEY,
                    object_id INTEGER,
                    prediction_time TIMESTAMP,
                    predicted_positions BLOB,
                    observed_positions BLOB,
                    corrections BLOB,
                    environmental_data BLOB,
                    model_type TEXT,
                    accuracy_score REAL,
                    FOREIGN KEY (object_id) REFERENCES multimodal_objects (id)
                )
            """)
            
            # Environmental data tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS environmental_data (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    source TEXT,
                    lake TEXT,
                    latitude REAL,
                    longitude REAL,
                    u_velocity REAL,
                    v_velocity REAL,
                    wind_u REAL,
                    wind_v REAL,
                    temperature REAL,
                    data_quality TEXT
                )
            """)
            
            # ML model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY,
                    model_name TEXT,
                    model_type TEXT,
                    training_date TIMESTAMP,
                    validation_score REAL,
                    test_score REAL,
                    parameters BLOB,
                    performance_metrics BLOB
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.installation_status['database_initialized'] = True
            logger.info(f"Database initialized at {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def create_configuration_files(self) -> bool:
        """Create necessary configuration files"""
        try:
            # Create enhanced config.yaml
            config_content = """
# CESAROPS Configuration File
# Enhanced with Multi-Modal ML capabilities

lakes:
  michigan:
    bounds: [-88.5, -85.5, 41.5, 46.0]
    erddap_url: "https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_MICHIGAN_3D.nc"
    glos_stations: ["obs_2", "obs_181", "obs_37"]
  erie:
    bounds: [-83.7, -78.8, 41.2, 42.9]
    erddap_url: "https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_ERIE_3D.nc"
  huron:
    bounds: [-85.0, -80.5, 43.2, 46.3]
    erddap_url: "https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_HURON_3D.nc"
  ontario:
    bounds: [-80.0, -76.0, 43.2, 44.3]
    erddap_url: "https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_ONTARIO_3D.nc"
  superior:
    bounds: [-92.5, -84.5, 46.0, 49.5]
    erddap_url: "https://coastwatch.glerl.noaa.gov/erddap/griddap/GLCFS_SUPERIOR_3D.nc"

# Multi-Modal ML Configuration
ml_config:
  sentence_transformer_model: "all-MiniLM-L6-v2"
  attention_heads: 4
  transformer_layers: 2
  hidden_dim: 64
  fcnn_layers: [128, 64, 32, 16]
  training_batch_size: 32
  learning_rate: 0.001
  
# Default simulation parameters
simulation:
  duration_hours: 24
  timestep_minutes: 10
  particles_per_hour: 60
  seed_radius_nm: 2.0
  default_windage: 0.03
  default_stokes: 0.01

# Rosa case validation parameters
rosa_case:
  coordinates: [42.995, -87.845]
  datetime: "2024-08-22 20:00:00"
  validated_windage: 0.06
  validated_stokes: 0.0045
  recovery_location: [42.403, -86.275]
  accuracy_threshold_nm: 25.0

# Data sources
data_sources:
  glerl_erddap: "https://coastwatch.glerl.noaa.gov/erddap/"
  glos_seagull: "https://seagull.glos.us/api/"
  ndbc_buoys: "https://www.ndbc.noaa.gov/data/realtime2/"
  noaa_gdp: "https://www.aoml.noaa.gov/phod/gdp/"
  
# File paths
paths:
  data_dir: "data"
  models_dir: "models"
  outputs_dir: "outputs"
  logs_dir: "logs"
"""
            
            config_path = self.base_dir / 'config.yaml'
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Create requirements.txt for easy installation
            requirements_content = """
# CESAROPS Requirements
# Core packages
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
requests>=2.25.0
netcdf4>=1.5.0
xarray>=0.19.0
pyyaml>=5.4.0

# Machine Learning
tensorflow>=2.8.0
scikit-learn>=1.0.0
joblib>=1.0.0
sentence-transformers>=2.2.0
transformers>=4.20.0
torch>=1.12.0

# Geospatial
cartopy>=0.20.0
geopandas>=0.10.0
shapely>=1.8.0
pyproj>=3.2.0
folium>=0.12.0

# Optional
jupyter>=1.0.0
ipykernel>=6.0.0
plotly>=5.0.0
dash>=2.0.0
opendrift>=1.8.0
"""
            
            requirements_path = self.base_dir / 'requirements.txt'
            with open(requirements_path, 'w') as f:
                f.write(requirements_content)
            
            # Create launch script for Windows
            if self.system == 'windows':
                launch_script = f"""@echo off
echo Starting CESAROPS Multi-Modal System...
call conda activate {self.conda_env_name}
if errorlevel 1 (
    echo Failed to activate conda environment
    pause
    exit /b 1
)

echo Environment activated successfully
python multimodal_fcnn_system.py
pause
"""
                launch_path = self.base_dir / 'launch_multimodal_cesarops.bat'
                with open(launch_path, 'w') as f:
                    f.write(launch_script)
            
            logger.info("Configuration files created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create configuration files: {e}")
            return False
    
    def run_diagnostic_tests(self) -> bool:
        """Run comprehensive diagnostic tests"""
        logger.info("Running diagnostic tests...")
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Python environment
        total_tests += 1
        try:
            result = subprocess.run([
                'conda', 'run', '-n', self.conda_env_name,
                'python', '-c', 'import sys; print(sys.version)'
            ], capture_output=True, text=True, check=True)
            logger.info(f"✓ Python version: {result.stdout.strip()}")
            tests_passed += 1
        except Exception as e:
            logger.error(f"✗ Python test failed: {e}")
        
        # Test 2: Core packages
        core_packages = ['numpy', 'pandas', 'matplotlib', 'requests']
        for package in core_packages:
            total_tests += 1
            try:
                subprocess.run([
                    'conda', 'run', '-n', self.conda_env_name,
                    'python', '-c', f'import {package}; print("{package} OK")'
                ], capture_output=True, text=True, check=True)
                logger.info(f"✓ {package} import successful")
                tests_passed += 1
            except Exception as e:
                logger.error(f"✗ {package} import failed: {e}")
        
        # Test 3: ML packages
        ml_packages = ['tensorflow', 'sklearn', 'sentence_transformers']
        for package in ml_packages:
            total_tests += 1
            try:
                if package == 'sklearn':
                    import_name = 'sklearn'
                elif package == 'sentence_transformers':
                    import_name = 'sentence_transformers'
                else:
                    import_name = package
                    
                subprocess.run([
                    'conda', 'run', '-n', self.conda_env_name,
                    'python', '-c', f'import {import_name}; print("{package} OK")'
                ], capture_output=True, text=True, check=True)
                logger.info(f"✓ {package} import successful")
                tests_passed += 1
            except Exception as e:
                logger.error(f"✗ {package} import failed: {e}")
        
        # Test 4: Database connectivity
        total_tests += 1
        try:
            db_path = self.base_dir / 'drift_objects.db'
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"✓ Database connectivity OK ({table_count} tables)")
            tests_passed += 1
        except Exception as e:
            logger.error(f"✗ Database test failed: {e}")
        
        # Test 5: Internet connectivity
        total_tests += 1
        try:
            urllib.request.urlopen('https://www.google.com', timeout=5)
            logger.info("✓ Internet connectivity OK")
            tests_passed += 1
        except Exception as e:
            logger.error(f"✗ Internet connectivity failed: {e}")
        
        success_rate = tests_passed / total_tests
        logger.info(f"Diagnostic tests completed: {tests_passed}/{total_tests} passed ({success_rate:.1%})")
        
        if success_rate >= 0.8:  # 80% success rate required
            self.installation_status['tests_passed'] = True
            return True
        else:
            return False
    
    def setup_ml_models(self) -> bool:
        """Set up and validate ML model infrastructure"""
        try:
            # Create models directory
            models_dir = self.base_dir / 'models'
            models_dir.mkdir(exist_ok=True)
            
            # Test TensorFlow installation
            logger.info("Testing TensorFlow installation...")
            result = subprocess.run([
                'conda', 'run', '-n', self.conda_env_name,
                'python', '-c', 
                'import tensorflow as tf; print(f"TensorFlow {tf.__version__} - GPU available: {tf.config.list_physical_devices(\"GPU\")}")'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✓ {result.stdout.strip()}")
            else:
                logger.warning(f"TensorFlow test warning: {result.stderr}")
            
            # Test Sentence Transformers
            logger.info("Testing Sentence Transformers...")
            result = subprocess.run([
                'conda', 'run', '-n', self.conda_env_name,
                'python', '-c', 
                'from sentence_transformers import SentenceTransformer; '
                'model = SentenceTransformer("all-MiniLM-L6-v2"); '
                'print("Sentence Transformers OK")'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✓ Sentence Transformers validation successful")
            else:
                logger.error(f"Sentence Transformers test failed: {result.stderr}")
                return False
            
            self.installation_status['ml_models_ready'] = True
            return True
            
        except Exception as e:
            logger.error(f"ML models setup failed: {e}")
            return False
    
    def generate_installation_report(self) -> str:
        """Generate comprehensive installation report"""
        report = []
        report.append("CESAROPS Installation Report")
        report.append("=" * 50)
        report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"System: {platform.system()} {platform.release()}")
        report.append(f"Python: {sys.version}")
        report.append("")
        
        report.append("Installation Status:")
        for component, status in self.installation_status.items():
            status_icon = "✓" if status else "✗"
            report.append(f"  {status_icon} {component.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
        
        report.append("")
        
        # Overall status
        all_passed = all(self.installation_status.values())
        if all_passed:
            report.append("🎉 INSTALLATION SUCCESSFUL!")
            report.append("")
            report.append("Next steps:")
            report.append(f"1. Activate environment: conda activate {self.conda_env_name}")
            report.append("2. Run Rosa case validation: python multimodal_fcnn_system.py")
            report.append("3. Check logs in: cesarops_installation.log")
        else:
            report.append("❌ INSTALLATION INCOMPLETE")
            report.append("")
            report.append("Failed components:")
            for component, status in self.installation_status.items():
                if not status:
                    report.append(f"  - {component.replace('_', ' ').title()}")
            
            report.append("")
            report.append("Troubleshooting:")
            report.append("1. Check internet connectivity")
            report.append("2. Verify conda installation")
            report.append("3. Review logs: cesarops_installation.log")
            report.append("4. Try manual package installation")
        
        return "\n".join(report)
    
    def run_complete_installation(self) -> bool:
        """Run complete installation process"""
        logger.info("Starting CESAROPS complete installation...")
        
        # Step 1: Check conda
        if not self.check_conda_installation():
            if not self.install_conda():
                logger.error("Conda installation failed")
                return False
        
        # Step 2: Create environment
        if not self.create_conda_environment():
            logger.error("Environment creation failed")
            return False
        
        # Step 3: Install dependencies
        if not self.install_dependencies():
            logger.error("Dependency installation failed")
            return False
        
        # Step 4: Initialize database
        if not self.initialize_database():
            logger.error("Database initialization failed")
            return False
        
        # Step 5: Create configuration files
        if not self.create_configuration_files():
            logger.error("Configuration file creation failed")
            return False
        
        # Step 6: Set up ML models
        if not self.setup_ml_models():
            logger.error("ML models setup failed")
            return False
        
        # Step 7: Run diagnostic tests
        if not self.run_diagnostic_tests():
            logger.warning("Some diagnostic tests failed, but installation may still be functional")
        
        # Generate report
        report = self.generate_installation_report()
        
        # Save report
        report_path = self.base_dir / 'installation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print("\n" + report)
        
        return all(self.installation_status.values())

def main():
    """Main installation function"""
    print("CESAROPS Multi-Modal Installation System")
    print("=" * 50)
    print("This will install all components for the state-of-the-art")
    print("SAR drift prediction system with ML enhancements.")
    print("")
    
    # Confirm installation
    response = input("Continue with installation? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        return
    
    installer = CESAROPSInstaller()
    success = installer.run_complete_installation()
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print(f"Activate environment with: conda activate {installer.conda_env_name}")
    else:
        print("\n❌ Installation completed with errors.")
        print("Check installation_report.txt and cesarops_installation.log for details.")
    
    return success

if __name__ == "__main__":
    main()