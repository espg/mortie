# Building Mortie with Rust Acceleration

This guide covers building mortie with its Rust-accelerated morton indexing functions.

## Prerequisites

### Required
- Python 3.10 or later
- Rust toolchain (rustc, cargo)
- Python packages: numpy, healpy

### Installing Rust

#### Linux/macOS
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### Windows
Download and run [rustup-init.exe](https://rustup.rs/)

### Verify Installation
```bash
rustc --version
cargo --version
```

## Development Build

For local development with Rust acceleration:

```bash
# Clone repository
git clone https://github.com/espg/mortie.git
cd mortie

# Install maturin (Rust-Python build tool)
pip install maturin

# Build and install in development mode
maturin develop --release

# Or for debugging with symbols
maturin develop
```

## Production Build

Build optimized wheels for distribution:

```bash
# Build wheel for current platform
maturin build --release

# Output will be in target/wheels/
ls -lh target/wheels/
```

## Testing

### Run tests with Rust implementation
```bash
pytest -v
```

### Run tests with numba fallback (for comparison)
```bash
MORTIE_FORCE_NUMBA=1 pytest -v
```

### Run Rust unit tests
```bash
cargo test
```

### Run benchmarks
```bash
cargo bench
```

## Installation from PyPI

Pre-built wheels are available for common platforms:

```bash
pip install mortie
```

This will automatically use the Rust implementation if a wheel is available for your platform.

## Fallback to Numba

If Rust is not available or compilation fails, mortie will automatically fall back to the numba implementation:

```bash
# Install with numba fallback
pip install mortie[numba]
```

## Platform-Specific Notes

### Linux
- Uses manylinux wheels for broad compatibility
- Supports x86_64 and aarch64 architectures

### macOS
- Separate wheels for Intel (x86_64) and Apple Silicon (aarch64)
- Minimum macOS version: 10.12

### Windows
- Requires Visual Studio Build Tools or equivalent
- Supports x86_64 architecture

## Build Options

### Release Build (Optimized)
```bash
maturin develop --release
```
- Full optimizations (opt-level = 3)
- Link-time optimization (LTO)
- Stripped binaries
- ~30-50% faster than debug builds

### Debug Build (Fast Compilation)
```bash
maturin develop
```
- Includes debug symbols
- Faster compilation
- Easier debugging with rust-gdb/rust-lldb

### Profile Build
```bash
maturin develop --profile profiling
```
- Optimized but with debug symbols
- Useful for performance profiling

## Troubleshooting

### "maturin: command not found"
```bash
pip install --upgrade maturin
```

### "Rust toolchain not found"
```bash
# Install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Or update existing installation
rustup update
```

### Build fails on Windows
Ensure you have Visual Studio Build Tools installed:
1. Download from: https://visualstudio.microsoft.com/downloads/
2. Install "Desktop development with C++"
3. Restart terminal and try again

### Import error: "cannot import name mortie_rs"
The Rust extension wasn't built. Run:
```bash
maturin develop --release
```

### Tests fail after rebuild
Clean build artifacts:
```bash
cargo clean
maturin develop --release
pytest -v
```

## Performance Comparison

Expected performance vs numba:

| Operation | Numba | Rust | Speedup |
|-----------|-------|------|---------|
| Single value | 10 µs | 5 µs | 2x |
| 1,000 values | 150 µs | 80 µs | 1.9x |
| 1M values | 145 ms | 75 ms | 1.9x |
| 1.2M (Antarctic) | 180 ms | 90 ms | 2x |

*Benchmarks on Intel i7, actual results may vary*

## CI/CD

GitHub Actions automatically builds wheels for:
- Linux (x86_64, aarch64)
- macOS (x86_64, aarch64)
- Windows (x86_64)
- Python 3.10, 3.11, 3.12, 3.13

See `.github/workflows/build-wheels.yml` for details.

## Contributing

When modifying Rust code:

1. Run Rust tests: `cargo test`
2. Run Python tests: `pytest -v`
3. Run benchmarks: `cargo bench`
4. Format code: `cargo fmt`
5. Check lints: `cargo clippy`

## Further Reading

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin User Guide](https://www.maturin.rs/)
- [Rust Book](https://doc.rust-lang.org/book/)
