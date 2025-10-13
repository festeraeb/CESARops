#!/usr/bin/env python3
"""
Check Training Data Summary
"""
import sqlite3

def check_data():
    conn = sqlite3.connect('drift_objects.db')
    cursor = conn.cursor()
    
    # Total data
    cursor.execute('SELECT COUNT(*), COUNT(DISTINCT drifter_id) FROM real_drifter_trajectories')
    total, drifters = cursor.fetchone()
    
    # By source
    cursor.execute('SELECT source, COUNT(*) FROM real_drifter_trajectories GROUP BY source')
    sources = cursor.fetchall()
    
    print(f"Training Data Summary:")
    print(f"  Total trajectory points: {total}")
    print(f"  Unique drifters: {drifters}")
    print(f"  By source:")
    for source, count in sources:
        print(f"    {source}: {count} points")
    
    conn.close()

if __name__ == "__main__":
    check_data()