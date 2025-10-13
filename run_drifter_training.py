#!/usr/bin/env python3
"""
Automated Drifter Training Script
=================================

Runs the complete drifter data collection and ML training pipeline
automatically without user prompts. Designed for batch processing
and automated scheduling.

Usage:
    python run_drifter_training.py [--quick] [--models-only]
    
Options:
    --quick: Collect data from last 30 days only (faster)
    --models-only: Skip data collection, train models on existing data
    --help: Show this help message

Author: GitHub Copilot
Date: January 7, 2025
License: MIT
"""

import sys
import argparse
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from drifter_training_pipeline import DrifterTrainingPipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Training pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

def run_quick_training():
    """Run training with minimal data collection (last 30 days)"""
    print("🚀 Running QUICK drifter training (30 days data)...")
    
    if not PIPELINE_AVAILABLE:
        print("❌ Training pipeline dependencies not available")
        print("Run: python comprehensive_installer.py")
        return False
    
    pipeline = DrifterTrainingPipeline()
    
    # Collect minimal data
    print("📊 Collecting recent data...")
    collection_results = pipeline.collect_all_data(max_workers=3)
    
    # Train models if we have data
    if collection_results.get('gdp_trajectories', 0) > 100:
        print("🧠 Training ML models...")
        training_results = pipeline.train_all_models()
        
        success = training_results.get('lstm_trained', False) or training_results.get('rf_trained', False)
        
        if success:
            print("✅ Quick training completed successfully!")
            print(f"   Training samples: {training_results.get('training_samples', 0)}")
            return True
        else:
            print("⚠️ Model training failed")
            return False
    else:
        print("❌ Insufficient data collected for training")
        return False

def run_models_only():
    """Train models on existing data without new collection"""
    print("🧠 Training models on existing data...")
    
    if not PIPELINE_AVAILABLE:
        print("❌ Training pipeline dependencies not available")
        return False
    
    pipeline = DrifterTrainingPipeline()
    training_results = pipeline.train_all_models()
    
    success = training_results.get('lstm_trained', False) or training_results.get('rf_trained', False)
    
    if success:
        print("✅ Model training completed!")
        print(f"   Training samples: {training_results.get('training_samples', 0)}")
        if training_results.get('lstm_trained'):
            print("   ✓ LSTM model saved")
        if training_results.get('rf_trained'):
            print("   ✓ Random Forest model saved")
        return True
    else:
        print("❌ Model training failed")
        for error in training_results.get('errors', []):
            print(f"      Error: {error}")
        return False

def run_full_training():
    """Run complete training pipeline"""
    print("🚀 Running FULL drifter training pipeline...")
    
    if not PIPELINE_AVAILABLE:
        print("❌ Training pipeline dependencies not available")
        print("Run: python comprehensive_installer.py")
        return False
    
    pipeline = DrifterTrainingPipeline()
    
    start_time = time.time()
    results = pipeline.run_complete_pipeline()
    end_time = time.time()
    
    print(f"\n⏱️ Pipeline completed in {end_time - start_time:.1f} seconds")
    
    if results.get('pipeline_success', False):
        print("🎉 Full training pipeline completed successfully!")
        
        # Print summary
        collection = results.get('collection', {})
        training = results.get('training', {})
        
        print(f"📊 Data collected:")
        print(f"   • GDP trajectories: {collection.get('gdp_trajectories', 0)} points")
        print(f"   • NDBC observations: {collection.get('ndbc_observations', 0)} points")
        print(f"   • GLOS observations: {collection.get('glos_observations', 0)} points")
        
        print(f"🧠 Models trained:")
        print(f"   • LSTM: {'✓' if training.get('lstm_trained') else '✗'}")
        print(f"   • Random Forest: {'✓' if training.get('rf_trained') else '✗'}")
        print(f"   • Training samples: {training.get('training_samples', 0)}")
        
        return True
    else:
        print("⚠️ Pipeline completed with issues")
        print("Check drifter_training.log for details")
        return False

def check_environment():
    """Check if the conda environment is properly set up"""
    try:
        import numpy
        import pandas
        import requests
        import sqlite3
        print("✅ Core dependencies available")
        
        try:
            import tensorflow
            import sklearn
            print("✅ ML dependencies available")
            return True
        except ImportError:
            print("⚠️ ML dependencies missing (tensorflow, scikit-learn)")
            print("   Models will not be trainable")
            return False
            
    except ImportError as e:
        print(f"❌ Core dependencies missing: {e}")
        print("   Run: python comprehensive_installer.py")
        return False

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Automated drifter data collection and ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_drifter_training.py                # Full training pipeline
    python run_drifter_training.py --quick        # Quick training (30 days)
    python run_drifter_training.py --models-only  # Train models on existing data
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick training with minimal data collection')
    parser.add_argument('--models-only', action='store_true',
                       help='Train models on existing data only')
    parser.add_argument('--check-env', action='store_true',
                       help='Check environment and dependencies')
    
    args = parser.parse_args()
    
    print("CESAROPS Automated Drifter Training")
    print("=" * 40)
    
    # Check environment first
    if args.check_env:
        env_ok = check_environment()
        return 0 if env_ok else 1
    
    # Check basic environment
    env_ok = check_environment()
    if not env_ok and not args.models_only:
        print("\n❌ Environment check failed")
        print("Install dependencies first: python comprehensive_installer.py")
        return 1
    
    # Run appropriate training mode
    success = False
    
    try:
        if args.models_only:
            success = run_models_only()
        elif args.quick:
            success = run_quick_training()
        else:
            success = run_full_training()
        
        if success:
            print(f"\n🎉 Training completed successfully!")
            print("Models saved to: models/")
            print("Data stored in: drift_objects.db")
            print("Report saved to: drifter_training_report.txt")
        else:
            print(f"\n❌ Training failed or incomplete")
            print("Check logs: drifter_training.log")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)