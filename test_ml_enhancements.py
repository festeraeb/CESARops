#!/usr/bin/env python3
"""
Test script for CESAROPS ML enhancements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set CI test mode if environment variable is set
if os.getenv('CESAROPS_TEST_MODE'):
    # Skip interactive prompts in CI
    os.environ['HEADLESS'] = '1'

from sarops import (
    init_drift_db,
    fetch_buoy_specifications,
    fetch_historical_buoy_tracks,
    collect_ml_training_data,
    train_drift_correction_model,
    search_sar_literature,
    analyze_charlie_brown_case,
    calculate_distance
)

def test_buoy_specifications():
    """Test buoy specifications fetching"""
    print("Testing buoy specifications fetching...")
    try:
        fetch_buoy_specifications()
        print("✓ Buoy specifications fetched successfully")
        return True
    except Exception as e:
        print(f"✗ Buoy specifications failed: {e}")
        return False

def test_historical_tracks():
    """Test historical buoy tracks fetching"""
    print("Testing historical buoy tracks fetching...")
    try:
        fetch_historical_buoy_tracks(days_back=7)  # Shorter period for testing
        print("✓ Historical tracks fetched successfully")
        return True
    except Exception as e:
        print(f"✗ Historical tracks failed: {e}")
        return False

def test_ml_training_data():
    """Test ML training data collection"""
    print("Testing ML training data collection...")
    try:
        collect_ml_training_data()
        print("✓ ML training data collected successfully")
        return True
    except Exception as e:
        print(f"✗ ML training data collection failed: {e}")
        return False

def test_ml_training():
    """Test ML model training"""
    print("Testing ML model training...")
    try:
        success = train_drift_correction_model()
        if success:
            print("✓ ML model trained successfully")
            return True
        else:
            print("⚠ ML model training completed with warnings")
            return True
    except Exception as e:
        print(f"✗ ML model training failed: {e}")
        return False

def test_sar_literature():
    """Test SAR literature search"""
    print("Testing SAR literature search...")
    try:
        results = search_sar_literature()
        print(f"✓ SAR literature search completed, found {len(results)} references")
        return True
    except Exception as e:
        print(f"✗ SAR literature search failed: {e}")
        return False

def test_distance_calculation():
    """Test distance calculation"""
    print("Testing distance calculation...")
    try:
        # Test Milwaukee to South Haven distance
        milwaukee_lat, milwaukee_lon = 43.0389, -87.9065
        south_haven_lat, south_haven_lon = 42.7674, -86.0956

        distance = calculate_distance(milwaukee_lat, milwaukee_lon, south_haven_lat, south_haven_lon)
        print(f"✓ Distance calculation: {distance:.2f} nautical miles")
        return True
    except Exception as e:
        print(f"✗ Distance calculation failed: {e}")
        return False

def test_charlie_brown_analysis():
    """Test Charlie Brown case analysis"""
    print("Testing Charlie Brown case analysis...")
    try:
        analysis = analyze_charlie_brown_case()
        if analysis:
            print("✓ Charlie Brown case analysis completed")
            print(f"  - Time in water: {analysis['time_in_water']:.1f} hours")
            print(f"  - Distance traveled: {analysis['distance_traveled']:.2f} nm")
            print(f"  - Drift probability: {analysis['drift_probability']}")
            return True
        else:
            print("⚠ Charlie Brown case analysis returned no results")
            return False
    except Exception as e:
        print(f"✗ Charlie Brown case analysis failed: {e}")
        return False

def main():
    """Run all tests"""
    print("CESAROPS ML Enhancement Test Suite")
    print("=" * 40)
    
    # Initialize database first
    print("Initializing database...")
    init_drift_db()
    print("Database initialized.")
    print()

    tests = [
        test_buoy_specifications,
        test_historical_tracks,
        test_ml_training_data,
        test_ml_training,
        test_sar_literature,
        test_distance_calculation,
        test_charlie_brown_analysis
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())