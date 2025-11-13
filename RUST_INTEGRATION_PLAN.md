# Rust Integration Plan for Mortie

## Overview
Replace numba-accelerated functions with Rust implementations to eliminate Dask distributed conflicts while maintaining or improving performance.

## Target Functions
Two functions with `@vectorize` decorators need Rust implementation:

### 1. `fastNorm2Mort(order, normed, parents)` - General version
- **Signature**: `int64(int64, int64, int64)`
- **Purpose**: Convert normalized HEALPix addresses to Morton indices
- **Constraints**: Order must be ≤18 (64-bit int limit)
- **Logic**: Bit manipulation loop with conditional parent cell handling

### 2. `VaexNorm2Mort(normed, parents)` - Order 18 specialized
- **Signature**: `int64(int64, int64)`
- **Purpose**: Same as fastNorm2Mort but order hardcoded to 18 for Vaex compatibility
- **Logic**: Identical to fastNorm2Mort(18, normed, parents)

## Technology Stack

### Rust Crate Structure
```
mortie-rs/          # Rust implementation
├── Cargo.toml      # Rust dependencies
├── src/
│   ├── lib.rs      # PyO3 bindings
│   └── morton.rs   # Core morton encoding logic
└── benches/        # Performance benchmarks
    └── morton_bench.rs
```

### Python Integration (PyO3 + Maturin)
- **PyO3**: Rust-Python bindings (https://pyo3.rs/)
- **Maturin**: Build tool for PyO3 wheels (https://www.maturin.rs/)
- **NumPy support**: PyO3 numpy integration for array handling

## Implementation Phases

### Phase 1: Project Setup ✓ (Current)
- [x] Create feature branch `feature/rust-acceleration`
- [ ] Initialize Rust crate with maturin
- [ ] Set up PyO3 dependencies
- [ ] Configure pyproject.toml for maturin builds

### Phase 2: Core Rust Implementation
- [ ] Implement `fast_norm2mort_scalar(order, normed, parents) -> i64`
- [ ] Implement `vaex_norm2mort_scalar(normed, parents) -> i64`
- [ ] Add input validation and error handling
- [ ] Write Rust unit tests

### Phase 3: NumPy Array Bindings
- [ ] Implement `fast_norm2mort(order, normed, parents) -> ndarray[i64]`
  - Handle scalar and array inputs via PyO3 numpy
  - Parallel iteration over arrays using rayon
- [ ] Implement `vaex_norm2mort(normed, parents) -> ndarray[i64]`
- [ ] Optimize for contiguous array access

### Phase 4: Python Integration
- [ ] Update `mortie/tools.py`:
  ```python
  try:
      from mortie_rs import fast_norm2mort, vaex_norm2mort
      RUST_AVAILABLE = True
  except ImportError:
      # Fall back to numba implementations
      RUST_AVAILABLE = False
      # Keep existing @vectorize implementations
  ```
- [ ] Create compatibility layer for seamless fallback

### Phase 5: Testing & Validation
- [ ] Run existing pytest suite (50 tests)
- [ ] Verify 1.2M+ coordinate regression test passes
- [ ] Compare Rust vs Numba outputs (should be identical)
- [ ] Performance benchmarking (Rust should be faster)
- [ ] Memory profiling

### Phase 6: Wheel Building & Distribution
- [ ] Configure maturin for multi-platform builds
- [ ] Set up GitHub Actions for wheel building:
  - Linux: x86_64, aarch64
  - macOS: x86_64, aarch64 (Apple Silicon)
  - Windows: x86_64
- [ ] Build wheels for Python 3.10, 3.11, 3.12, 3.13
- [ ] Test wheels on clean environments
- [ ] Configure PyPI release workflow

## Build Configuration

### pyproject.toml Updates
```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
dependencies = [
  "numpy>=1.20",
  "healpy",
  # numba and dependencies become optional
]

[project.optional-dependencies]
numba = ["numba", "cython", "llvmlite"]

[tool.maturin]
python-source = "."
module-name = "mortie._mortie_rs"
bindings = "pyo3"
features = ["pyo3/extension-module"]
```

### Cargo.toml
```toml
[package]
name = "mortie-rs"
version = "0.1.0"
edition = "2021"

[lib]
name = "mortie_rs"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
numpy = "0.21"
rayon = "1.8"  # Parallel iteration

[dev-dependencies]
criterion = "0.5"  # Benchmarking
```

## Testing Strategy

### 1. Unit Tests (Rust)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_norm2mort_basic() {
        let result = fast_norm2mort_scalar(18, 100, 2);
        assert_eq!(result, /* expected value */);
    }

    #[test]
    fn test_parent_branch_negative() {
        // Test parent >= 6 branch (negative result)
        let result = fast_norm2mort_scalar(18, 100, 8);
        assert!(result < 0);
    }
}
```

### 2. Integration Tests (Python)
- All existing 50 tests must pass without modification
- Regression test on 1.2M Antarctic coordinates must match exactly
- Add explicit Rust vs Numba comparison tests

### 3. Performance Benchmarks
```rust
// benches/morton_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_fast_norm2mort(c: &mut Criterion) {
    c.bench_function("fast_norm2mort_1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                fast_norm2mort_scalar(black_box(18), black_box(i), black_box(2));
            }
        });
    });
}
```

## CI/CD Workflow

### GitHub Actions: `build-wheels.yml`
```yaml
name: Build Wheels

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build-wheels:
    name: Build wheels on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-13, macos-14]

    steps:
    - uses: actions/checkout@v4

    - name: Build wheels
      uses: PyO3/maturin-action@v1
      with:
        command: build
        args: --release --out dist
        manylinux: auto

    - name: Upload wheels
      uses: actions/upload-artifact@v4
      with:
        name: wheels-${{ matrix.os }}
        path: dist
```

## Performance Expectations

### Numba (Current)
- JIT compilation overhead on first call
- Threading conflicts with Dask workers
- ~50-100ms for 1.2M coordinates (after JIT)

### Rust (Target)
- No JIT overhead (AOT compiled)
- No threading conflicts (pure functions)
- Expected: 30-80ms for 1.2M coordinates
- Lower memory footprint

## Migration Path

### Development (Immediate)
```python
pip install maturin
cd mortie
maturin develop  # Build and install in development mode
pytest -v  # Run tests
```

### Production (After merge)
```python
pip install mortie  # Gets pre-built wheel with Rust
# Numba becomes optional dependency for legacy support
```

## Rollback Strategy
- Keep numba implementations in codebase
- Runtime detection: Use Rust if available, fallback to numba
- Feature flag for testing: `MORTIE_FORCE_NUMBA=1`

## Documentation Updates
- [ ] README: Add Rust build requirements
- [ ] Installation guide for developers (Rust toolchain)
- [ ] Performance comparison section
- [ ] Migration notes for existing users

## Success Criteria
1. ✅ All 50 unit tests pass
2. ✅ 1.2M coordinate regression test passes (exact match)
3. ✅ Wheels build on all platforms (Linux, macOS, Windows)
4. ✅ Performance ≥ numba (ideally faster)
5. ✅ No Dask distributed conflicts
6. ✅ PyPI release includes pre-built wheels

## Timeline Estimate
- Phase 1-2: 2-3 hours (Setup + Core Rust)
- Phase 3-4: 2-3 hours (Bindings + Integration)
- Phase 5: 1-2 hours (Testing)
- Phase 6: 2-3 hours (CI/CD + Wheels)
- **Total**: 7-11 hours of focused development
