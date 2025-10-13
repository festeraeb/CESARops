#!/usr/bin/env python3
"""
Test the enhanced ML pipeline step by step
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sarops import (
    fetch_gdp_drifter_tracks, 
    fetch_enhanced_environmental_data, 
    collect_ml_training_data,
    train_drift_correction_model
)
import sqlite3

def test_ml_pipeline():
    db_file = 'test_ml.db'
    
    # Clean start
    if os.path.exists(db_file):
        os.remove(db_file)
    
    print("🧪 Testing Enhanced ML Pipeline")
    print("=" * 40)
    
    # Step 1: Fetch drifter tracks
    print("Step 1: Fetching drifter tracks...")
    try:
        fetch_gdp_drifter_tracks(db_file)
        print("✅ Drifter tracks fetched")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Step 2: Fetch environmental data
    print("\nStep 2: Fetching environmental data...")
    try:
        fetch_enhanced_environmental_data(db_file)
        print("✅ Environmental data fetched")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Step 3: Check database contents
    print("\nStep 3: Checking database contents...")
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        
        # List tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in c.fetchall()]
        print(f"Tables created: {tables}")
        
        # Count records in each table
        for table in ['drifter_tracks', 'environmental_conditions']:
            if table in tables:
                c.execute(f'SELECT COUNT(*) FROM {table}')
                count = c.fetchone()[0]
                print(f"  {table}: {count} records")
            else:
                print(f"  {table}: ❌ TABLE MISSING")
        
        conn.close()
        print("✅ Database check complete")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    # Step 4: Test ML training data collection
    print("\nStep 4: Collecting ML training data...")
    try:
        success = collect_ml_training_data(db_file)
        if success:
            print("✅ ML training data collected")
        else:
            print("⚠️  ML training data collection had issues")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Step 5: Test model training
    print("\nStep 5: Training ML model...")
    try:
        success = train_drift_correction_model(db_file)
        if success:
            print("✅ ML model trained successfully")
        else:
            print("⚠️  ML model training had issues")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print("\n🎉 All tests completed!")
    return True

if __name__ == "__main__":
    test_ml_pipeline()