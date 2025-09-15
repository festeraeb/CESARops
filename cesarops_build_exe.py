#!/usr/bin/env python3
"""
Build standalone executable for CESAROPS
This script uses PyInstaller to create a single-file executable
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required build dependencies are installed"""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("✗ PyInstaller not found")
        print("Install with: pip install pyinstaller")
        return False
    
    # Check main script exists
    if not Path("cesarops_enhanced.py").exists():
        print("✗ cesarops_enhanced.py not found")
        return False
    
    print("✓ Main script found")
    return True

def create_spec_file():
    """Create PyInstaller spec file with custom configuration"""
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Define data files to include
data_files = [
    ('config.yaml', '.'),
]

# Define hidden imports (modules not automatically detected)
hidden_imports = [
    'scipy.spatial.transform._rotation_groups',
    'sklearn.ensemble._forest',
    'sklearn.tree._utils',
    'joblib.format_stack',
    'yaml.loader',
    'yaml.dumper',
]

a = Analysis(
    ['cesarops_enhanced.py'],
    pathex=[],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Exclude if not using plotting
        'plotly',      # Exclude if not using web plotting
        'IPython',     # Not needed for GUI app
        'jupyter',     # Not needed for GUI app
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CESAROPS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want console window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='cesarops_icon.ico' if os.path.exists('cesarops_icon.ico') else None,
)

# Optional: Create a directory distribution as well
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CESAROPS_dist',
)
'''
    
    with open('cesarops.spec', 'w') as f:
        f.write(spec_content)
    
    print("✓ PyInstaller spec file created")

def clean_build_dirs():
    """Clean previous build directories"""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
            print(f"✓ Cleaned {dir_name}/")

def build_executable():
    """Build the executable using PyInstaller"""
    print("\nBuilding executable...")
    
    # Run PyInstaller
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean',  # Clean cache
        'cesarops.spec'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Build completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def create_installer_package():
    """Create a complete installer package"""
    
    # Create installer directory
    installer_dir = Path("CESAROPS_Installer")
    installer_dir.mkdir(exist_ok=True)
    
    # Copy executable
    if Path("dist/CESAROPS.exe").exists():
        shutil.copy2("dist/CESAROPS.exe", installer_dir / "CESAROPS.exe")
        print("✓ Executable copied to installer")
    else:
        print("✗ Executable not found")
        return False
    
    # Copy essential files
    essential_files = [
        "README.md",
        "config.yaml",
        "requirements.txt",
    ]
    
    for file_name in essential_files:
        if Path(file_name).exists():
            shutil.copy2(file_name, installer_dir / file_name)
            print(f"✓ Copied {file_name}")
    
    # Create directories
    for dir_name in ["data", "outputs", "models", "logs"]:
        (installer_dir / dir_name).mkdir(exist_ok=True)
        # Create .gitkeep files
        (installer_dir / dir_name / ".gitkeep").touch()
    
    print("✓ Created directory structure")
    
    # Create launcher batch file
    launcher_content = '''@echo off
setlocal
cd /d "%~dp0"

echo Starting CESAROPS...
echo.

:: Check if this is first run
if not exist "data\.initialized" (
    echo First-time setup...
    echo Initializing data directories...
    echo. > data\.initialized
    echo Setup complete!
    echo.
)

:: Launch CESAROPS
CESAROPS.exe

:: Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo CESAROPS exited with an error.
    echo Check logs\ directory for details.
    pause
)
'''
    
    with open(installer_dir / "Launch_CESAROPS.bat", 'w') as f:
        f.write(launcher_content)
    
    print("✓ Created launcher script")
    
    # Create installation instructions
    install_instructions = '''# CESAROPS Installation Instructions

## Quick Start
1. Extract this folder to your desired location
2. Double-click "Launch_CESAROPS.bat" to start

## System Requirements
- Windows 7 or later
- At least 4 GB RAM
- Internet connection for data fetching
- 1 GB free disk space

## First Run
- The first time you run CESAROPS, it will initialize its data directories
- An internet connection is recommended for fetching current data
- The application can work offline using cached data

## Troubleshooting
- If the application doesn't start, check the logs/ folder for error messages
- For Windows Defender/antivirus warnings, add CESAROPS.exe to exceptions
- For "missing DLL" errors, install Microsoft Visual C++ Redistributable

## Support
- For help and documentation, see README.md
- For technical issues, check the project repository
- This software is provided free for SAR operations

## Files Included
- CESAROPS.exe - Main application
- Launch_CESAROPS.bat - Easy launcher
- config.yaml - Configuration settings
- README.md - Full documentation
- requirements.txt - Python dependencies (for reference)
- data/ - Cached data storage
- outputs/ - Results output folder
- models/ - Machine learning models
- logs/ - Application logs
'''
    
    with open(installer_dir / "INSTALLATION.txt", 'w') as f:
        f.write(install_instructions)
    
    print("✓ Created installation instructions")
    
    return True

def create_zip_distribution():
    """Create ZIP file for distribution"""
    installer_dir = Path("CESAROPS_Installer")
    if not installer_dir.exists():
        print("✗ Installer directory not found")
        return False
    
    import zipfile
    
    zip_name = "CESAROPS_Portable.zip"
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in installer_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(installer_dir.parent)
                zip_file.write(file_path, arcname)
    
    print(f"✓ Created distribution: {zip_name}")
    
    # Get file size
    size_mb = Path(zip_name).stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")
    
    return True

def main():
    """Main build process"""
    print("CESAROPS Executable Builder")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("cesarops_enhanced.py").exists():
        print("✗ Must run from CESAROPS root directory")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Create default config if it doesn't exist
    if not Path("config.yaml").exists():
        print("Creating default config...")
        import yaml
        default_config = {
            'erddap': {
                'lmhofs': 'https://coastwatch.glerl.noaa.gov/erddap',
                'rtofs': 'https://coastwatch.pfeg.noaa.gov/erddap',
                'hycom': 'https://tds.hycom.org/erddap'
            },
            'drift_defaults': {
                'dt_minutes': 10,
                'duration_hours': 24,
                'windage': 0.03,
                'stokes': 0.01
            },
            'seeding': {
                'default_radius_nm': 2.0,
                'default_rate': 60
            }
        }
        with open('config.yaml', 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    # Build process
    steps = [
        ("Cleaning build directories", clean_build_dirs),
        ("Creating spec file", create_spec_file),
        ("Building executable", build_executable),
        ("Creating installer package", create_installer_package),
        ("Creating ZIP distribution", create_zip_distribution),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"✗ Failed: {step_name}")
            return 1
    
    print("\n" + "=" * 40)
    print("✓ Build completed successfully!")
    print("\nFiles created:")
    print("- dist/CESAROPS.exe (standalone executable)")
    print("- CESAROPS_Installer/ (complete package)")
    print("- CESAROPS_Portable.zip (distribution archive)")
    
    print("\nNext steps:")
    print("1. Test the executable: dist/CESAROPS.exe")
    print("2. Test the installer: CESAROPS_Installer/Launch_CESAROPS.bat")
    print("3. Distribute: CESAROPS_Portable.zip")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
