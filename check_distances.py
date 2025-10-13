#!/usr/bin/env python3
"""
Quick analysis to see what distances we're getting in Rosa forward seeding
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_fast_engine import SimpleFastDriftEngine
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DISTANCE_CHECK')

def test_single_scenario():
    """Test one scenario to see what distance we get"""
    
    engine = SimpleFastDriftEngine()
    
    # Test scenario: 10nm off Milwaukee at 90° bearing, released at 8 PM Aug 22
    release_lat = 43.1389  # ~10nm off Milwaukee
    release_lon = -87.7565
    release_time = datetime(2025, 8, 22, 20, 0, 0)  # 8 PM
    target_time = datetime(2025, 8, 28, 16, 0, 0)   # Aug 28, 4 PM
    
    duration_hours = (target_time - release_time).total_seconds() / 3600
    
    logger.info(f"Testing scenario: {duration_hours:.1f} hours drift")
    logger.info(f"Release: {release_lat:.4f}°N, {release_lon:.4f}°W")
    
    # Create environmental conditions
    env_conditions = []
    num_steps = int(duration_hours / 0.5) + 1
    
    for i in range(num_steps):
        time_offset = i * 1800  # 30 minutes in seconds
        env_conditions.append({
            'timestamp': release_time + timedelta(seconds=time_offset),
            'wind_speed': 3.1,  # From real GLOS data
            'wind_direction': 93,  # From real GLOS data  
            'current_u': 0.0,
            'current_v': 0.0,
            'wave_height': 1.0
        })
    
    # Run simulation
    results = engine.simulate_drift_ensemble(
        release_lat=release_lat,
        release_lon=release_lon,
        release_time=release_time,
        duration_hours=duration_hours,
        environmental_data=env_conditions,
        n_particles=50,
        object_specs={
            'windage': 0.06,      # User's parameter
            'leeway': 0.06,       
            'stokes_factor': 0.0045  # User's parameter
        }
    )
    
    # Calculate distance to South Haven
    center_position = results['statistics']['center_position']
    final_lat = center_position['lat']
    final_lon = center_position['lon']
    
    target_pos = (42.40, -86.28)  # South Haven recovery area
    final_pos = (final_lat, final_lon)
    distance_nm = geodesic(target_pos, final_pos).nautical
    
    logger.info(f"Final position: {final_lat:.4f}°N, {final_lon:.4f}°W")
    logger.info(f"Distance to South Haven: {distance_nm:.1f} nm")
    logger.info(f"Target was: 42.40°N, -86.28°W")
    
    # Check what the distance spread looks like
    distance_stats = results['statistics']['distance_stats']
    logger.info(f"Particle spread: {distance_stats['std_nm']:.1f} nm standard deviation")
    logger.info(f"95th percentile: {distance_stats['percentile_95_nm']:.1f} nm from center")
    
    return distance_nm

if __name__ == "__main__":
    distance = test_single_scenario()