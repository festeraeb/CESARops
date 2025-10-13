#!/usr/bin/env python3
"""
Quick re-analysis of Rosa forward seeding with more realistic tolerance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rosa_forward_seeding import RosaForwardSeeding
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ROSA_QUICK')

def quick_distance_check():
    """Run a few scenarios to check actual distances"""
    
    logger.info("🔍 Quick distance check for Rosa forward seeding")
    
    analyzer = RosaForwardSeeding()
    
    # Test just a few scenarios to see distances
    test_points = [
        {'lat': 43.1389, 'lon': -87.7565, 'name': '10nm at 90°', 'bearing': 90, 'distance_nm': 10},  
        {'lat': 43.0889, 'lon': -87.7265, 'name': '8nm at 90°', 'bearing': 90, 'distance_nm': 8},   
        {'lat': 43.0389, 'lon': -87.6965, 'name': '6nm at 90°', 'bearing': 90, 'distance_nm': 6},   
    ]
    
    test_times = [
        {'hour': 18, 'name': '6 PM'},
        {'hour': 20, 'name': '8 PM'},
        {'hour': 22, 'name': '10 PM'}
    ]
    
    logger.info(f"Testing {len(test_points)} points × {len(test_times)} times = {len(test_points) * len(test_times)} scenarios")
    
    results = []
    
    for point in test_points:
        for time_info in test_times:
            
            release_time_obj = {
                'time': analyzer.target_recovery['time'].replace(
                    year=2025, month=8, day=22, 
                    hour=time_info['hour'], minute=0, second=0
                ),
                'duration_hours': None,
                'date_str': f"Aug 22 {time_info['hour']:02d}:00",
                'hours_before': None
            }
            
            # Calculate duration
            duration = analyzer.target_recovery['time'] - release_time_obj['time']
            release_time_obj['duration_hours'] = duration.total_seconds() / 3600
            release_time_obj['hours_before'] = release_time_obj['duration_hours']
            
            logger.info(f"🧪 Testing: {point['name']} at {time_info['name']} ({release_time_obj['duration_hours']:.1f}h drift)")
            
            # Run single scenario
            result = analyzer.run_scenario(point, release_time_obj)
            
            if result and 'error' not in result:
                distance = result['final_distance_nm']
                logger.info(f"   📍 Distance to South Haven: {distance:.1f} nm")
                
                results.append({
                    'point': point['name'],
                    'time': time_info['name'],
                    'distance_nm': distance,
                    'position': f"{result['final_lat']:.3f}°N, {result['final_lon']:.3f}°W"
                })
            else:
                logger.warning(f"   ❌ Simulation failed")
    
    # Summary
    logger.info("\n📊 DISTANCE SUMMARY:")
    logger.info("=" * 50)
    
    if results:
        distances = [r['distance_nm'] for r in results]
        min_dist = min(distances)
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        
        logger.info(f"Closest: {min_dist:.1f} nm")
        logger.info(f"Farthest: {max_dist:.1f} nm") 
        logger.info(f"Average: {avg_dist:.1f} nm")
        
        # Show best scenarios
        results.sort(key=lambda x: x['distance_nm'])
        logger.info("\n🏆 BEST SCENARIOS:")
        for i, r in enumerate(results[:3]):
            logger.info(f"{i+1}. {r['point']} at {r['time']}: {r['distance_nm']:.1f} nm")
            logger.info(f"    Final: {r['position']}")
    
    return results

if __name__ == "__main__":
    results = quick_distance_check()