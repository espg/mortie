# Building Mortie with Rust Acceleration

This guide covers building mortie with its Rust-accelerated morton indexing functions.

## Prerequisites

### Required
- Python 3.10 or later
- Rust toolchain (rustc, cargo)
- Python packages: numpy

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

## Workspace Layout

The Rust side is a cargo workspace with two members:

| path | crate | contents |
|---|---|---|
| `./` (root manifest) | `mortie`, library `mortie_rustie` | the pyo3 extension: geometry, coverage/MOC, the rayon batch kernels, the Arrow FFI, and every `#[pyfunction]`. Sources in `src_rust/`, benches in `src_rust/benches/`. |
| `mortie-core/` | `mortie-core` | the packed-word codec only: the bit layout, `encode`/`decode`, order/truncation/containment arithmetic, the decimal-string grammar, and the `(depth, nested-ipix) ↔ packed-word` pivot primitives. |

`mortie-core` has **no dependencies** — not pyo3, not numpy, not rayon, not a
HEALPix crate — so a Rust project can depend on the codec alone. That is a
contract, not an accident: `mortie-core/tests/dep_contract.rs` fails the suite if
the crate's manifest ever declares a dependency table.

`mortie_rustie` depends on `mortie-core` and re-exports it, so
`mortie_rustie::decimal_morton` and `mortie_rustie::morton` resolve exactly as
they did before the split, and nothing about the Python surface changes. The
wheel is still built from the root manifest by maturin, which pulls the path
dependency into the build (and into the sdist) automatically.

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

### Run Rust unit tests
```bash
# Both workspace members
cargo test

# The codec crate on its own (no pyo3, so no Python needed to link)
cargo test -p mortie-core
```

`cargo test` on the root package needs the extension-module symbols resolved
lazily on *every* platform, since pyo3's `extension-module` feature deliberately
skips linking libpython. The invocation below is the **macOS** remedy only:
`-undefined dynamic_lookup` is a Mach-O linker option, so it does nothing on
Linux or Windows — those need their own equivalent, which is not documented here
because it has not been verified against this workspace.

```bash
# macOS
RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo test
```

`cargo test -p mortie-core` needs no flag on any platform — the codec crate
links nothing.

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

### Import error: "cannot import name '_rustie'"
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

Performance comparison of Rust vs Python (reference) implementations:

| Benchmark | Rust | Python (reference) | Speedup |
|-----------|------|--------------------|---------|
| Scalar operations | 0.14 µs | 10.69 µs | **78.6x** |
| Small arrays (1K) | 1.93 ms | 4.14 ms | **2.1x** |
| Large arrays (100K) | 1.85 ms | 410.59 ms | **222.2x** |
| Real-world (1.2M coords) | 102.51 ms | 5109.15 ms | **49.8x** |

The Rust implementation provides dramatic performance improvements, especially for large datasets.

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
