#!/usr/bin/env python3
"""
CESAROPS Quick Start Launcher
============================

Simple launcher that guides users through the installation and setup process
for the complete CESAROPS multi-modal SAR system.

Author: GitHub Copilot
Date: January 7, 2025
License: MIT
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("""
   ╔═══════════════════════════════════════════════════════════════╗
   ║              CESAROPS Multi-Modal SAR System                  ║
   ║                    Quick Start Launcher                       ║
   ╠═══════════════════════════════════════════════════════════════╣
   ║  State-of-the-art Search and Rescue drift prediction with:   ║
   ║  • FCNN dynamic correction with GPS feedback                 ║
   ║  • Multi-modal Sentence Transformer descriptions             ║
   ║  • Attention-based trajectory prediction                     ║
   ║  • Variable immersion ratio modeling                         ║
   ║  • Rosa fender case validation (24.3 nm accuracy)            ║
   ╚═══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   CESAROPS requires Python 3.8 or higher")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} (suitable)")
    return True

def check_conda():
    """Check if conda is available"""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
            return True
        else:
            print("❌ Conda command failed")
            return False
    except FileNotFoundError:
        print("❌ Conda not found")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'comprehensive_installer.py',
        'multimodal_fcnn_system.py',
        'rosa_multimodal_demo.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    else:
        print("✓ All required files present")
        return True

def run_installer():
    """Run the comprehensive installer"""
    print("\n🚀 Starting installation process...")
    print("This will:")
    print("  • Install/check conda environment")
    print("  • Install all Python dependencies")
    print("  • Set up database and configuration")
    print("  • Run diagnostic tests")
    print("  • Validate ML components")
    
    response = input("\nProceed with installation? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Installation cancelled")
        return False
    
    try:
        result = subprocess.run([sys.executable, 'comprehensive_installer.py'], 
                              check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print("❌ Installation failed - check logs for details")
        return False

def run_rosa_demo():
    """Run the Rosa case demonstration"""
    print("\n🎯 Running Rosa fender case demonstration...")
    print("This will demonstrate:")
    print("  • Multi-modal object description")
    print("  • FCNN dynamic trajectory correction")
    print("  • Validation against 24.3 nm accuracy")
    print("  • Comparison with hand calculations")
    
    try:
        # Try to run in conda environment if available
        conda_env = "cesarops"
        result = subprocess.run([
            'conda', 'run', '-n', conda_env, 
            'python', 'rosa_multimodal_demo.py'
        ], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        print("❌ Demo failed - environment may not be set up")
        print("Try running the installer first")
        return False
    except FileNotFoundError:
        # Fall back to direct Python execution
        try:
            result = subprocess.run([sys.executable, 'rosa_multimodal_demo.py'], 
                                  check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            print("❌ Demo failed - dependencies may not be installed")
            return False

def show_menu():
    """Show main menu"""
    while True:
        print("\n" + "="*50)
        print("CESAROPS Quick Start Menu")
        print("="*50)
        print("1. Run system checks")
        print("2. Install CESAROPS system")
        print("3. Run Rosa case demonstration")
        print("4. Activate conda environment")
        print("5. Show system information")
        print("6. Exit")
        print("="*50)
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            run_system_checks()
        elif choice == '2':
            run_installer()
        elif choice == '3':
            run_rosa_demo()
        elif choice == '4':
            show_conda_activation()
        elif choice == '5':
            show_system_info()
        elif choice == '6':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please select 1-6.")

def run_system_checks():
    """Run comprehensive system checks"""
    print("\n🔍 Running system checks...")
    print("-" * 30)
    
    checks_passed = 0
    total_checks = 4
    
    # Python version check
    if check_python_version():
        checks_passed += 1
    
    # Conda check
    if check_conda():
        checks_passed += 1
    
    # File check
    if check_files():
        checks_passed += 1
    
    # Environment check
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        if 'cesarops' in result.stdout:
            print("✓ CESAROPS environment exists")
            checks_passed += 1
        else:
            print("❌ CESAROPS environment not found")
    except:
        print("❌ Cannot check conda environments")
    
    print(f"\nSystem Status: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 System ready! You can run the Rosa demonstration.")
    elif checks_passed >= 2:
        print("⚠️  System partially ready. Consider running the installer.")
    else:
        print("❌ System not ready. Please install dependencies first.")

def show_conda_activation():
    """Show how to activate conda environment"""
    print("\n📋 Conda Environment Activation")
    print("-" * 35)
    print("To manually activate the CESAROPS environment:")
    print("")
    
    if platform.system().lower() == 'windows':
        print("  conda activate cesarops")
        print("  python rosa_multimodal_demo.py")
    else:
        print("  conda activate cesarops")
        print("  python rosa_multimodal_demo.py")
    
    print("\nTo deactivate:")
    print("  conda deactivate")

def show_system_info():
    """Show system information"""
    print("\n💻 System Information")
    print("-" * 25)
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check disk space
    try:
        if platform.system().lower() == 'windows':
            import shutil
            total, used, free = shutil.disk_usage('.')
            print(f"Available Disk Space: {free // (2**30)} GB")
    except:
        pass
    
    # Check conda environments
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        envs = [line.strip() for line in result.stdout.split('\n') 
                if line.strip() and not line.startswith('#')]
        print(f"Conda Environments: {len(envs)} found")
        if 'cesarops' in result.stdout:
            print("  • CESAROPS environment: ✓ Available")
        else:
            print("  • CESAROPS environment: ❌ Not found")
    except:
        print("Conda: Not available")

def main():
    """Main launcher function"""
    print_banner()
    
    # Quick system check
    print("Performing quick system check...")
    python_ok = check_python_version()
    conda_ok = check_conda()
    files_ok = check_files()
    
    if not python_ok:
        print("\n❌ Python version incompatible. Please upgrade and try again.")
        return
    
    if not files_ok:
        print("\n❌ Required files missing. Please check installation.")
        return
    
    if python_ok and conda_ok and files_ok:
        print("\n✅ Basic requirements met!")
        
        # Check if already installed
        try:
            result = subprocess.run(['conda', 'env', 'list'], 
                                  capture_output=True, text=True)
            if 'cesarops' in result.stdout:
                print("🎯 CESAROPS environment detected - ready to run!")
                
                response = input("\nRun Rosa demonstration now? (Y/n): ").strip().lower()
                if response in ['', 'y', 'yes']:
                    if run_rosa_demo():
                        print("\n🎉 Demonstration completed successfully!")
                    return
        except:
            pass
    
    # Show menu for manual control
    show_menu()

if __name__ == "__main__":
    main()