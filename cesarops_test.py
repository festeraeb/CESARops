#!/usr/bin/env python3
"""
CESAROPS System Test and Validation
Run this to verify your installation is working correctly
"""

import sys
import os
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    required_modules = [
        ('sys', 'System'),
        ('os', 'Operating System'),
        ('datetime', 'Date/Time'),
        ('pathlib', 'Path handling'),
        ('sqlite3', 'Database'),
        ('logging', 'Logging'),
        ('threading', 'Threading'),
        ('queue', 'Queue'),
        ('tkinter', 'GUI Framework'),
    ]
    
    core_modules = [
        ('pandas', 'Data Analysis'),
        ('numpy', 'Numerical Computing'),
        ('requests', 'HTTP Requests'),
        ('scipy.spatial', 'Spatial Computing'),
        ('yaml', 'YAML Configuration'),
    ]
    
    optional_modules = [
        ('sklearn', 'Machine Learning'),
        ('joblib', 'ML Persistence'),
        ('simplekml', 'KML Export'),
        ('matplotlib', 'Plotting'),
    ]
    
    # Test required modules
    failed_required = []
    for module, desc in required_modules:
        try:
            __import__(module)
            print(f"  ✓ {desc}")
        except ImportError as e:
            print(f"  ✗ {desc}: {e}")
            failed_required.append(module)
    
    # Test core modules
    failed_core = []
    for module, desc in core_modules:
        try:
            __import__(module)
            print(f"  ✓ {desc}")
        except ImportError as e:
            print(f"  ✗ {desc}: {e}")
            failed_core.append(module)
    
    # Test optional modules
    available_optional = []
    for module, desc in optional_modules:
        try:
            __import__(module)
            print(f"  ✓ {desc} (optional)")
            available_optional.append(module)
        except ImportError:
            print(f"  - {desc} (optional, not installed)")
    
    return len(failed_required) == 0 and len(failed_core) == 0, available_optional

def test_file_structure():
    """Test required file structure"""
    print("\nTesting file structure...")
    
    required_files = [
        'cesarops_enhanced.py',
        'config.yaml',
        'requirements.txt',
    ]
    
    required_dirs = [
        'data',
        'outputs', 
        'models',
    ]
    
    missing_files = []
    for filename in required_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (missing)")
            missing_files.append(filename)
    
    missing_dirs = []
    for dirname in required_dirs:
        dir_path = Path(dirname)
        if dir_path.exists():
            print(f"  ✓ {dirname}/ directory")
        else:
            print(f"  - {dirname}/ directory (creating)")
            dir_path.mkdir(exist_ok=True)
    
    return len(missing_files) == 0

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        import yaml
        
        if not Path('config.yaml').exists():
            print("  - Creating default config.yaml")
            default_config = {
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
                yaml.dump(default_config, f, default_flow_style=False)
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate config structure
        required_keys = ['erddap', 'drift_defaults', 'seeding']
        for key in required_keys:
            if key in config:
                print(f"  ✓ Config section: {key}")
            else:
                print(f"  ✗ Missing config section: {key}")
                return False
        
        print("  ✓ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\nTesting database...")
    
    try:
        import sqlite3
        
        # Test database creation
        test_db = "test_cesarops.db"
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_table (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                value REAL
            )
        ''')
        
        # Insert test data
        cursor.execute("INSERT INTO test_table (timestamp, value) VALUES (?, ?)", 
                      (datetime.now().isoformat(), 123.45))
        
        # Read test data
        cursor.execute("SELECT * FROM test_table")
        rows = cursor.fetchall()
        
        conn.commit()
        conn.close()
        
        # Clean up
        os.remove(test_db)
        
        print(f"  ✓ Database operations successful ({len(rows)} records)")
        return True
        
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False

def test_data_processing():
    """Test core data processing functionality"""
    print("\nTesting data processing...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
            'lat': [42.0 + i * 0.1 for i in range(5)],
            'lon': [-86.0 - i * 0.1 for i in range(5)],
            'u': [0.1 * i for i in range(5)],
            'v': [0.05 * i for i in range(5)]
        })
        
        print(f"  ✓ Created test dataset ({len(test_data)} records)")
        
        # Test basic operations
        mean_lat = test_data['lat'].mean()
        print(f"  ✓ Data analysis (mean lat: {mean_lat:.3f})")
        
        # Test scipy functionality
        from scipy.spatial import cKDTree
        points = test_data[['lat', 'lon']].values
        tree = cKDTree(points)
        distances, indices = tree.query([42.1, -86.1], k=2)
        print(f"  ✓ Spatial indexing (nearest distance: {distances[0]:.4f})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data processing error: {e}")
        traceback.print_exc()
        return False

def test_gui_components():
    """Test GUI components"""
    print("\nTesting GUI components...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Create test window (don't show it)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test basic widgets
        frame = ttk.Frame(root)
        label = ttk.Label(frame, text="Test")
        entry = ttk.Entry(frame)
        button = ttk.Button(frame, text="Test")
        
        print("  ✓ Basic GUI widgets")
        
        # Test variables
        str_var = tk.StringVar(value="test")
        bool_var = tk.BooleanVar(value=True)
        double_var = tk.DoubleVar(value=3.14)
        
        print("  ✓ GUI variables")
        
        root.destroy()
        print("  ✓ GUI framework operational")
        
        return True
        
    except Exception as e:
        print(f"  ✗ GUI error: {e}")
        return False

def test_network_connectivity():
    """Test network connectivity to data sources"""
    print("\nTesting network connectivity...")
    
    try:
        import requests
        
        test_urls = [
            ('NOAA GLERL', 'https://coastwatch.glerl.noaa.gov/erddap/info/index.html'),
            ('NOAA PFEG', 'https://coastwatch.pfeg.noaa.gov/erddap/info/index.html'),
        ]
        
        session = requests.Session()
        session.headers.update({'User-Agent': 'CESAROPS-Test/1.0'})
        
        connectivity_ok = True
        for name, url in test_urls:
            try:
                response = session.head(url, timeout=10)
                if response.status_code == 200:
                    print(f"  ✓ {name} (accessible)")
                else:
                    print(f"  ? {name} (status: {response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"  ✗ {name} (error: {type(e).__name__})")
                connectivity_ok = False
        
        return connectivity_ok
        
    except Exception as e:
        print(f"  ✗ Network test error: {e}")
        return False

def test_drift_calculations():
    """Test basic drift calculations"""
    print("\nTesting drift calculations...")
    
    try:
        import numpy as np
        
        # Test coordinate conversion
        lat, lon = 42.0, -86.0
        u_ms, v_ms = 0.1, 0.05  # m/s
        dt_s = 600  # 10 minutes
        
        # Simple drift calculation
        R_earth = 6371000.0
        dlat = (v_ms * dt_s / R_earth) * (180.0 / np.pi)
        dlon = (u_ms * dt_s / (R_earth * np.cos(np.radians(lat)))) * (180.0 / np.pi)
        
        new_lat = lat + dlat
        new_lon = lon + dlon
        
        print(f"  ✓ Coordinate transformation ({lat:.6f},{lon:.6f}) -> ({new_lat:.6f},{new_lon:.6f})")
        
        # Test vector operations
        angles = np.linspace(0, 2*np.pi, 10)
        x_coords = np.cos(angles)
        y_coords = np.sin(angles)
        
        print(f"  ✓ Vector operations ({len(angles)} points)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Drift calculation error: {e}")
        return False

def generate_system_report():
    """Generate system information report"""
    print("\nSystem Information:")
    print("-" * 30)
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working Directory: {os.getcwd()}")
    
    try:
        import pandas as pd
        print(f"Pandas Version: {pd.__version__}")
    except ImportError:
        print("Pandas: Not installed")
    
    try:
        import numpy as np
        print(f"NumPy Version: {np.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    try:
        import requests
        print(f"Requests Version: {requests.__version__}")
    except ImportError:
        print("Requests: Not installed")
    
    try:
        import scipy
        print(f"SciPy Version: {scipy.__version__}")
    except ImportError:
        print("SciPy: Not installed")

def main():
    """Main test runner"""
    print("CESAROPS System Test & Validation")
    print("=" * 50)
    
    # Run all tests
    tests = [
        ("Import Dependencies", test_imports),
        ("File Structure", test_file_structure),
        ("Configuration", test_configuration),
        ("Database Operations", test_database),
        ("Data Processing", test_data_processing),
        ("GUI Components", test_gui_components),
        ("Network Connectivity", test_network_connectivity),
        ("Drift Calculations", test_drift_calculations),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  ✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Generate system report
    generate_system_report()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✓" if success else "✗"
        print(f"{icon} {test_name}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! CESAROPS should work correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")
        print("\nCommon solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Check internet connection for network tests")
        print("- Verify file permissions in the project directory")
        return 1

if __name__ == "__main__":
    sys.exit(main())
