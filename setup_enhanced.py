#!/usr/bin/env python3
"""
Build script for CESAROPS Rust extensions
"""

import subprocess
import sys
import os
from pathlib import Path

def build_rust_extension():
    """Build the Rust extension module"""
    
    print("🦀 Building Rust high-performance core...")
    
    rust_dir = Path("rust_core")
    
    if not rust_dir.exists():
        print("❌ Rust source directory not found")
        return False
    
    try:
        # Check if Rust is installed
        result = subprocess.run(["cargo", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Rust/Cargo not found. Please install Rust from https://rustup.rs/")
            return False
        
        print(f"✅ Found Rust: {result.stdout.strip()}")
        
        # Install maturin if not available
        try:
            subprocess.run(["maturin", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📦 Installing maturin...")
            subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
        
        # Build the extension
        print("🔨 Building extension...")
        os.chdir(rust_dir)
        
        result = subprocess.run([
            "maturin", "develop", "--release"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Rust extension built successfully!")
            return True
        else:
            print(f"❌ Build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error building Rust extension: {e}")
        return False
    finally:
        os.chdir("..")

def install_dependencies():
    """Install required Python dependencies"""
    
    print("📦 Installing Python dependencies...")
    
    dependencies = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0",
        "matplotlib>=3.5.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    print("✅ Dependencies installed successfully!")
    return True

def setup_development_environment():
    """Set up the complete development environment"""
    
    print("🚀 Setting up CESAROPS Enhanced Development Environment")
    print("=" * 60)
    
    # Install Python dependencies
    if not install_dependencies():
        print("❌ Failed to install Python dependencies")
        return False
    
    # Try to build Rust extension (optional)
    rust_success = build_rust_extension()
    
    if rust_success:
        print("✅ High-performance Rust core available")
    else:
        print("⚠️  Rust core not available, will use Python fallback")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python enhanced_drift_analysis.py")
    print("2. Or run: python test_enhanced_ml.py")
    
    return True

if __name__ == "__main__":
    setup_development_environment()