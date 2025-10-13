#!/usr/bin/env python3
"""
Check Database Contents for Rosa Case Analysis
"""
import sqlite3
from datetime import datetime

def check_database():
    conn = sqlite3.connect('drift_objects.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print("Available Tables:")
    print("=" * 20)
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} records")
    
    print("\nEnvironmental Data Around Rosa Case Timeframe:")
    print("=" * 50)
    
    # Check for environmental data around August 22-23, 2025
    if 'environmental_conditions' in tables:
        cursor.execute("""
            SELECT timestamp, latitude, longitude, wind_speed, wind_direction, 
                   current_u, current_v, water_temp 
            FROM environmental_conditions 
            WHERE timestamp LIKE '2025-08%' 
            ORDER BY timestamp
        """)
        
        env_data = cursor.fetchall()
        print(f"Environmental data records: {len(env_data)}")
        if env_data:
            for record in env_data[:5]:  # Show first 5
                print(f"  {record}")
    
    # Check for wind data
    if 'wind_data' in tables:
        cursor.execute("SELECT COUNT(*) FROM wind_data WHERE timestamp LIKE '2025-08%'")
        wind_count = cursor.fetchone()[0]
        print(f"Wind data records for August 2025: {wind_count}")
    
    # Check for current data
    if 'current_data' in tables:
        cursor.execute("SELECT COUNT(*) FROM current_data WHERE timestamp LIKE '2025-08%'")
        current_count = cursor.fetchone()[0]
        print(f"Current data records for August 2025: {current_count}")
    
    # Check for wave data
    if 'wave_data' in tables:
        cursor.execute("SELECT COUNT(*) FROM wave_data WHERE timestamp LIKE '2025-08%'")
        wave_count = cursor.fetchone()[0]
        print(f"Wave data records for August 2025: {wave_count}")
    
    print("\nRosa Case Details:")
    print("=" * 18)
    print("Last known location: ~42.995°N, 87.845°W (off Milwaukee)")
    print("Incident time: August 22, 2025 ~8:00 PM")
    print("Found location: South Haven, MI (42.4030°N, 86.2750°W)")
    print("Found time: August 23, 2025 morning")
    print("Estimated drift time: ~12 hours")
    
    conn.close()

if __name__ == "__main__":
    check_database()