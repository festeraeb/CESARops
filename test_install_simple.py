#!/usr/bin/env python3
"""
CESAROPS Installation Test - Simple Version
Quick test to verify your installation is working
"""

import sys
print(f"Python version: {sys.version}")
print("=" * 50)

# Test core packages
core_packages = [
    ("pandas", "Data processing"),
    ("numpy", "Numerical computing"), 
    ("requests", "HTTP requests"),
    ("yaml", "Configuration files"),
    ("scipy", "Scientific computing")
]

print("Testing CORE packages (required):")
core_ok = True
for package, description in core_packages:
    try:
        __import__(package)
        print(f"✓ {package:12} - {description}")
    except ImportError:
        print(f"✗ {package:12} - {description} - MISSING")
        core_ok = False

print("\nTesting GUI support:")
try:
    import tkinter as tk
    # Test creating window
    root = tk.Tk()
    root.withdraw()  # Hide it
    root.destroy()
    print("✓ tkinter      - GUI framework - OK")
    gui_ok = True
except ImportError:
    print("✗ tkinter      - GUI framework - MISSING")
    gui_ok = False
except Exception as e:
    print(f"? tkinter      - GUI framework - ERROR: {e}")
    gui_ok = False

print("\nTesting OPTIONAL packages:")
optional_packages = [
    ("sklearn", "Machine Learning"),
    ("joblib", "ML model storage"),
    ("simplekml", "KML export"),
]

optional_count = 0
for package, description in optional_packages:
    try:
        __import__(package)
        print(f"✓ {package:12} - {description}")
        optional_count += 1
    except ImportError:
        print(f"- {package:12} - {description} - Not installed (optional)")

print("\n" + "=" * 50)
print("INSTALLATION TEST RESULTS:")
print("=" * 50)

if core_ok and gui_ok:
    print("🎉 EXCELLENT: Full installation working!")
    print("   You can run the complete CESAROPS system")
elif core_ok:
    print("✅ GOOD: Core functionality working!")
    print("   You can run CESAROPS (GUI might have issues)")
    print("   Consider reinstalling Python with tkinter support")
else:
    print("❌ PROBLEMS: Missing core packages")
    print("   Run: pip install pandas numpy requests PyYAML scipy")

print(f"\nOptional features available: {optional_count}/3")

# Test basic functionality
if core_ok:
    print("\nTesting basic operations...")
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # Test data creation
        data = pd.DataFrame({
            'lat': [42.0, 42.1, 42.2],
            'lon': [-86.0, -86.1, -86.2],
            'time': [datetime.now()] * 3
        })
        
        # Test basic operations
        mean_lat = data['lat'].mean()
        
        print(f"✓ Data processing test passed (mean lat: {mean_lat})")
        
        # Test scipy
        from scipy.spatial import distance
        coords = np.array([[42.0, -86.0], [42.1, -86.1]])
        dist = distance.euclidean(coords[0], coords[1])
        print(f"✓ Scientific computing test passed (distance: {dist:.4f})")
        
    except Exception as e:
        print(f"✗ Basic operations test failed: {e}")

print("\n" + "=" * 50)

# Recommendations
if not core_ok:
    print("NEXT STEPS:")
    print("1. Fix missing core packages:")
    print("   pip install pandas numpy requests PyYAML scipy")
    print("2. Re-run this test")
    print("3. Try CESAROPS after core packages work")
elif not gui_ok:
    print("NEXT STEPS:")  
    print("1. Try running CESAROPS anyway (might work)")
    print("2. If GUI fails, reinstall Python with tkinter")
    print("3. Consider using command-line mode")
else:
    print("READY TO GO!")
    print("1. Run: python cesarops_enhanced.py")
    print("2. Or double-click: run_cesarops.bat")
    print("3. Start with the Data Sources tab")

print("=" * 50)

# Quick config test
print("\nTesting configuration...")
try:
    import yaml
    
    test_config = {
        'erddap': {'lmhofs': 'https://coastwatch.glerl.noaa.gov/erddap'},
        'drift_defaults': {'dt_minutes': 10}
    }
    
    yaml_str = yaml.dump(test_config)
    loaded_config = yaml.safe_load(yaml_str)
    
    print("✓ Configuration system working")
    
except Exception as e:
    print(f"✗ Configuration test failed: {e}")

print("\nTest complete!")

# Keep window open on Windows
if sys.platform.startswith('win'):
    input("\nPress Enter to exit...")
