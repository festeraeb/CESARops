@echo off
setlocal enabledelayedexpansion

echo ================================================================
echo CESAROPS - Civilian Emergency SAR Operations - Installation
echo ================================================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://python.org
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

:: Check Python version using separate script
python check_python_version.py
if errorlevel 1 (
    echo ERROR: Python 3.8 or later is required
    echo Current version: %PYTHON_VERSION%
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing core dependencies...
pip install pandas numpy requests PyYAML scipy

echo.
echo Installing GUI support...
:: Check tkinter using separate script
python check_tkinter.py >nul 2>&1
if errorlevel 1 (
    echo WARNING: tkinter not available - GUI may not work
    echo You may need to reinstall Python with tkinter support
)

echo.
echo Installing optional dependencies...

:: Ask user about ML support
set /p INSTALL_ML=Install Machine Learning support? (y/n, default=y): 
if /i "%INSTALL_ML%"=="" set INSTALL_ML=y
if /i "%INSTALL_ML%"=="y" (
    echo Installing ML libraries...
    pip install scikit-learn joblib
) else (
    echo Skipping ML libraries
)

:: Ask about KML support
set /p INSTALL_KML=Install KML export support? (y/n, default=y): 
if /i "%INSTALL_KML%"=="" set INSTALL_KML=y
if /i "%INSTALL_KML%"=="y" (
    echo Installing KML support...
    pip install simplekml
) else (
    echo Skipping KML support
)

:: Ask about plotting
set /p INSTALL_PLOT=Install enhanced plotting? (y/n, default=n): 
if /i "%INSTALL_PLOT%"=="y" (
    echo Installing plotting libraries...
    pip install matplotlib plotly
) else (
    echo Skipping plotting libraries
)

echo.
echo Creating directories...
if not exist data mkdir data
if not exist outputs mkdir outputs
if not exist models mkdir models
if not exist logs mkdir logs

echo.
echo Creating default configuration...
if not exist config.yaml (
    python create_config.py > config_output.txt 2>&1
    type config_output.txt
    if errorlevel 1 (
        echo ERROR: Failed to create config.yaml
        type config_output.txt
        pause
        exit /b 1
    )
    del config_output.txt
) else (
    echo Configuration file already exists
)

echo.
echo Creating run script...
python create_run_script.py
if errorlevel 1 (
    echo ERROR: Failed to create run script
    pause
    exit /b 1
)

echo.
echo Testing installation...
python test_installation.py
if errorlevel 1 (
    echo.
    echo ERROR: Installation test failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Installation completed successfully!
echo ================================================================
echo.
echo To run CESAROPS:
echo   1. Double-click run_cesarops.bat
echo   OR
echo   2. Open command prompt in this folder and run: run_cesarops.bat
echo.
echo Files created:
echo   - venv/           (Python virtual environment)
echo   - config.yaml     (Configuration file)
echo   - run_cesarops.bat (Launch script)
echo   - data/           (Data storage)
echo   - outputs/        (Results output)
echo   - models/         (ML models)
echo.
echo For support, check the documentation or visit the project repository.
echo.
pause
