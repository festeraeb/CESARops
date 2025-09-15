@echo off
echo ================================================================
echo CESAROPS Quick Install - Core Components Only
echo ================================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.8+ first.
    pause
    exit /b 1
)

echo Creating virtual environment...
if exist venv rmdir /s /q venv
python -m venv venv
call venv\Scripts\activate.bat

echo Installing core dependencies only...
python -m pip install --upgrade pip

echo Installing essential packages...
pip install pandas numpy requests PyYAML scipy --no-cache-dir

echo Testing core installation...
python -c "
import pandas, numpy, requests, yaml, scipy
print('✓ Core packages installed successfully')
"

if errorlevel 1 (
    echo Core installation failed
    pause
    exit /b 1
)

echo Installing GUI support...
python -c "import tkinter; print('✓ GUI support available')" 2>nul
if errorlevel 1 (
    echo WARNING: tkinter not available - you may need to reinstall Python with tkinter
)

echo.
echo ================================================================
echo CORE INSTALLATION COMPLETE
echo ================================================================
echo.

set /p INSTALL_ML=Install Machine Learning support? (y/n, default=n): 
if /i "%INSTALL_ML%"=="y" (
    echo Installing ML libraries...
    pip install scikit-learn joblib --no-cache-dir
    if errorlevel 1 (
        echo ML installation failed - continuing without ML support
    ) else (
        echo ✓ ML support installed
    )
)

set /p INSTALL_KML=Install KML export? (y/n, default=y): 
if /i "%INSTALL_KML%"=="" set INSTALL_KML=y
if /i "%INSTALL_KML%"=="y" (
    echo Installing KML support...
    pip install simplekml --no-cache-dir
    if errorlevel 1 (
        echo KML installation failed - continuing without KML export
    ) else (
        echo ✓ KML support installed
    )
)

echo.
echo SKIPPING optional plotting libraries (matplotlib/plotly)
echo These can cause installation issues and aren't required for core functionality
echo.

echo Creating directories...
if not exist data mkdir data
if not exist outputs mkdir outputs
if not exist models mkdir models
if not exist logs mkdir logs

echo Creating run script...
echo @echo off > run_cesarops.bat
echo cd /d "%%~dp0" >> run_cesarops.bat
echo call venv\Scripts\activate.bat >> run_cesarops.bat
echo python cesarops_enhanced.py >> run_cesarops.bat
echo pause >> run_cesarops.bat

echo Creating minimal config...
(
echo erddap:
echo   lmhofs: "https://coastwatch.glerl.noaa.gov/erddap"
echo   rtofs: "https://coastwatch.pfeg.noaa.gov/erddap" 
echo   hycom: "https://tds.hycom.org/erddap"
echo drift_defaults:
echo   dt_minutes: 10
echo   duration_hours: 24
echo   windage: 0.03
echo   stokes: 0.01
echo seeding:
echo   default_radius_nm: 2.0
echo   default_rate: 60
) > config.yaml

echo.
echo ================================================================
echo QUICK INSTALL COMPLETE!
echo ================================================================
echo.
echo Ready to run:
echo   1. Double-click run_cesarops.bat
echo   2. Or run: python cesarops_enhanced.py
echo.
echo If you need plotting later, install manually:
echo   pip install matplotlib plotly
echo.
pause
