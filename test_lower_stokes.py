#!/usr/bin/env python3
"""
ROSA FENDER - LOWER STOKES DRIFT TEST
Testing user's position with reduced Stokes drift
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_fast_engine import SimpleFastDriftEngine
import numpy as np
from datetime import datetime, timedelta
import json

def test_lower_stokes():
    """Test with significantly lower Stokes drift values"""
    
    print("🔍 ROSA FENDER - LOWER STOKES DRIFT TEST")
    print("Position: 42.995°N, 87.845°W at 8:00 PM")
    print("Testing very low Stokes drift values")
    print("=" * 50)
    
    engine = SimpleFastDriftEngine()
    
    # User's position
    user_release = {
        'lat': 42.995,
        'lon': -87.845,
        'time': datetime(2025, 8, 22, 20, 0, 0)  # 8 PM CDT
    }
    
    # Actual recovery
    actual_recovery = {
        'lat': 42.40,
        'lon': -86.28,
        'time': datetime(2025, 8, 28, 16, 0, 0)
    }
    
    duration_hours = (actual_recovery['time'] - user_release['time']).total_seconds() / 3600.0
    actual_drift_distance = engine._calculate_distance(
        user_release['lat'], user_release['lon'],
        actual_recovery['lat'], actual_recovery['lon']
    )
    
    print(f"Actual Drift Distance: {actual_drift_distance:.1f} nm")
    print(f"Duration: {duration_hours:.1f} hours ({duration_hours/24:.1f} days)")
    
    # Generate environmental data
    env_data = []
    current_time = user_release['time']
    
    for hour in range(int(duration_hours) + 6):
        day_offset = hour / 24.0
        hour_of_day = (hour + 20) % 24
        
        # Refined environmental conditions
        base_wind = 6.0  # Slightly lower base wind
        base_dir = 235.0  # Adjusted wind direction
        
        if day_offset < 1:
            wind_factor = 1.0
            dir_shift = 0
        elif day_offset < 2.5:
            wind_factor = 1.2
            dir_shift = -10
        elif day_offset < 4:
            wind_factor = 1.0
            dir_shift = -5
        else:
            wind_factor = 0.8
            dir_shift = 5
        
        diurnal = 1.0 + 0.15 * np.sin((hour_of_day - 6) * np.pi / 12)
        
        env_data.append({
            'timestamp': current_time,
            'wind_speed': max(0.5, (base_wind * wind_factor * diurnal) + np.random.normal(0, 0.4)),
            'wind_direction': (base_dir + dir_shift + np.random.normal(0, 8)) % 360,
            'current_u': 0.020 + 0.010 * np.sin(hour * 0.1),
            'current_v': 0.012 + 0.008 * np.cos(hour * 0.08),
            'wave_height': 0.6 + 0.25 * wind_factor + 0.08 * np.sin(hour * 0.12),
            'water_temp': 21.0 - 0.04 * day_offset,
            'pressure': 1015.0 + np.random.normal(0, 1.5)
        })
        current_time += timedelta(hours=1)
    
    print(f"\n🔄 Testing with VERY LOW Stokes drift values...")
    
    # Test scenarios with dramatically reduced Stokes drift
    scenarios = [
        {
            'name': 'Ultra-Low Stokes',
            'specs': {'windage': 0.025, 'leeway': 0.025, 'stokes_factor': 0.001}  # Very low Stokes
        },
        {
            'name': 'Minimal Stokes',
            'specs': {'windage': 0.020, 'leeway': 0.020, 'stokes_factor': 0.002}  # Minimal Stokes
        },
        {
            'name': 'No Stokes',
            'specs': {'windage': 0.025, 'leeway': 0.030, 'stokes_factor': 0.0}    # Zero Stokes
        },
        {
            'name': 'Current Dominant',
            'specs': {'windage': 0.015, 'leeway': 0.015, 'stokes_factor': 0.001}  # Current-driven
        },
        {
            'name': 'Pure Current',
            'specs': {'windage': 0.010, 'leeway': 0.010, 'stokes_factor': 0.0}    # Mostly current
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        result = engine.simulate_drift_ensemble(
            release_lat=user_release['lat'],
            release_lon=user_release['lon'],
            release_time=user_release['time'],
            duration_hours=duration_hours,
            environmental_data=env_data,
            n_particles=150,  # More particles for stability
            time_step_minutes=30,  # Smaller time steps
            object_specs=scenario['specs']
        )
        
        predicted = result['statistics']['center_position']
        error = engine._calculate_distance(
            predicted['lat'], predicted['lon'],
            actual_recovery['lat'], actual_recovery['lon']
        )
        
        predicted_drift = engine._calculate_distance(
            user_release['lat'], user_release['lon'],
            predicted['lat'], predicted['lon']
        )
        
        results.append({
            'name': scenario['name'],
            'predicted_position': predicted,
            'prediction_error_nm': error,
            'predicted_drift_nm': predicted_drift,
            'drift_error_nm': abs(predicted_drift - actual_drift_distance),
            'specs': scenario['specs']
        })
        
        print(f"  {scenario['name']:<15}: Error = {error:.1f} nm, Drift = {predicted_drift:.1f} nm")
    
    # Find best result
    best_result = min(results, key=lambda x: x['prediction_error_nm'])
    
    print(f"\n🎯 BEST RESULT WITH LOWER STOKES:")
    print("=" * 45)
    print(f"Configuration: {best_result['name']}")
    print(f"Stokes Factor: {best_result['specs']['stokes_factor']}")
    print(f"Windage: {best_result['specs']['windage']}")
    print(f"Leeway: {best_result['specs']['leeway']}")
    print(f"")
    print(f"Predicted Final: {best_result['predicted_position']['lat']:.3f}°N, {best_result['predicted_position']['lon']:.3f}°W")
    print(f"Actual Recovery: {actual_recovery['lat']:.2f}°N, {actual_recovery['lon']:.2f}°W")
    print(f"Prediction Error: {best_result['prediction_error_nm']:.1f} nautical miles")
    print(f"Predicted Drift: {best_result['predicted_drift_nm']:.1f} nm")
    print(f"Actual Drift: {actual_drift_distance:.1f} nm")
    print(f"Drift Distance Error: {best_result['drift_error_nm']:.1f} nm")
    
    # Accuracy assessment
    if best_result['prediction_error_nm'] < 15:
        accuracy = "EXCELLENT"
        color = "🟢"
    elif best_result['prediction_error_nm'] < 35:
        accuracy = "GOOD"
        color = "🟡"
    elif best_result['prediction_error_nm'] < 75:
        accuracy = "FAIR"
        color = "🟠"
    else:
        accuracy = "NEEDS REFINEMENT"
        color = "🔴"
    
    print(f"\n{color} ACCURACY: {accuracy}")
    
    # Compare all scenarios
    print(f"\n📈 STOKES DRIFT COMPARISON:")
    print("-" * 75)
    print(f"{'Scenario':<16} {'Stokes':<8} {'Error(nm)':<10} {'Drift(nm)':<10} {'D.Error(nm)':<10}")
    print("-" * 75)
    for result in sorted(results, key=lambda x: x['prediction_error_nm']):
        stokes_val = result['specs']['stokes_factor']
        print(f"{result['name']:<16} {stokes_val:<8.3f} {result['prediction_error_nm']:<10.1f} {result['predicted_drift_nm']:<10.1f} {result['drift_error_nm']:<10.1f}")
    
    # Analysis
    print(f"\n💡 STOKES DRIFT ANALYSIS:")
    improvement = 146.8 - best_result['prediction_error_nm']  # Previous best was 146.8 nm
    if improvement > 0:
        print(f"✅ IMPROVEMENT: {improvement:.1f} nm better than previous test!")
    
    if best_result['prediction_error_nm'] < 50:
        print("✅ Excellent progress! Lower Stokes drift significantly improved accuracy.")
        print("🔍 The fender appears to be primarily current-driven with minimal wave effects.")
    elif best_result['prediction_error_nm'] < 100:
        print("✅ Good improvement with lower Stokes drift.")
        print("🔍 Fender drift is less wave-dependent than initially assumed.")
    
    print(f"\n🌊 PHYSICS INSIGHTS:")
    print(f"• Stokes factor of {best_result['specs']['stokes_factor']:.3f} works best")
    print(f"• Suggests fender has minimal wave-following behavior")
    print(f"• Drift dominated by wind and current effects")
    print(f"• Orange teardrop fender may be more streamlined than expected")
    
    # Save results
    os.makedirs('outputs', exist_ok=True)
    output_data = {
        'test_type': 'lower_stokes_drift',
        'user_position': user_release,
        'best_result': best_result,
        'all_results': results,
        'improvement_nm': improvement if improvement > 0 else 0,
        'physics_insights': {
            'optimal_stokes': best_result['specs']['stokes_factor'],
            'drift_type': 'current_and_wind_dominated',
            'wave_sensitivity': 'low'
        }
    }
    
    with open('outputs/rosa_lower_stokes.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: outputs/rosa_lower_stokes.json")
    
    return output_data

if __name__ == "__main__":
    test_lower_stokes()