#!/usr/bin/env python3
"""
Simple CI test script for CESAROPS integrated components
"""

import os
import sys

# Set CI test mode
os.environ['CESAROPS_TEST_MODE'] = '1'
os.environ['HEADLESS'] = '1'

def test_imports():
    """Test that core modules can be imported"""
    print("Testing core imports...")
    try:
        # Test basic imports
        import numpy
        import pandas
        import matplotlib
        import requests
        print("✓ Core scientific libraries imported successfully")
        
        # Test CESAROPS modules
        import sarops
        import cesarops
        print("✓ CESAROPS modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_database_init():
    """Test database initialization"""
    print("Testing database initialization...")
    try:
        from sarops import init_drift_db
        init_drift_db()
        print("✓ Database initialization successful")
        return True
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic CESAROPS functionality"""
    print("Testing basic functionality...")
    try:
        from sarops import calculate_distance
        
        # Test distance calculation
        distance = calculate_distance(42.0, -87.0, 43.0, -86.0)
        if distance > 0:
            print(f"✓ Distance calculation: {distance:.2f} nm")
            return True
        else:
            print("✗ Distance calculation returned invalid result")
            return False
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CESAROPS CI Test Suite")
    print("=" * 30)
    
    tests = [
        test_imports,
        test_database_init,
        test_basic_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 30)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    # Exit with proper code for CI
    if failed == 0:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()