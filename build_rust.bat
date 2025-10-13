@echo off
echo Installing geopy and building Rust core...
call conda activate cesarops
echo Current environment: %CONDA_DEFAULT_ENV%

echo Installing geopy...
pip install geopy

echo Current directory: %CD%
echo Available maturin: 
where maturin

echo Building Rust core...
cd rust_core
maturin develop --release

echo Done!
pause