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

### Run tests with pure Python fallback (for comparison)
```bash
MORTIE_FORCE_PYTHON=1 pytest -v
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

## Fallback to Pure Python

If Rust is not available or compilation fails, mortie will automatically fall back to a pure Python implementation:

```bash
# Install without Rust (pure Python fallback)
pip install mortie --no-binary mortie
```

The pure Python implementation is maintained as a reference and provides identical results, though with lower performance (~50x slower).

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

Performance comparison of Rust vs pure Python implementations:

| Benchmark | Rust | Pure Python | Speedup |
|-----------|------|-------------|---------|
| Scalar operations | 0.14 µs | 10.69 µs | **78.6x** |
| Small arrays (1K) | 1.93 ms | 4.14 ms | **2.1x** |
| Large arrays (100K) | 1.85 ms | 410.59 ms | **222.2x** |
| Real-world (1.2M coords) | 102.51 ms | 5109.15 ms | **49.8x** |

The Rust implementation provides dramatic performance improvements, especially for large datasets. The pure Python fallback is maintained as a reference implementation and provides identical results.

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
