@echo off
call conda activate cesarops
cd rust_core
echo Building Rust extension...
maturin develop --release
cd ..
echo Build complete!