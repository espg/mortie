# Rust Implementation Testing & Integration Plan

## Overview
This document outlines the testing strategy for the Rust-accelerated morton indexing implementation.

## Testing Layers

### 1. Rust Unit Tests (src_rust/src/morton.rs)
**Purpose**: Verify core Rust logic in isolation

**Tests Implemented**:
- ✅ `test_fast_norm2mort_basic` - Basic conversion functionality
- ✅ `test_fast_norm2mort_southern_hemisphere` - Parent >= 6 (negative results)
- ✅ `test_fast_norm2mort_northern_hemisphere` - Parent < 6 (positive results)
- ✅ `test_fast_norm2mort_order_too_large` - Order validation (max 18)
- ✅ `test_fast_norm2mort_order_18` - Maximum order support
- ✅ `test_fast_norm2mort_deterministic` - Reproducibility
- ✅ `test_fast_norm2mort_all_parents` - All parent cells (0-11)
- ✅ `test_powers_of_10` - Precomputed constants validation
- ✅ `test_powers_of_4` - Precomputed constants validation

**Run with**:
```bash
cargo test
cargo test -- --nocapture  # With output
```

### 2. Python Integration Tests (mortie/tests/)
**Purpose**: Verify Rust implementation matches numba behavior exactly

**Existing Test Suite** (50 tests):
- `test_tools.py` - 39 unit tests covering all functions
- `test_polygon_regression.py` - 10 tests on 1.2M real coordinates
- `test_morton_counts.py` - 1 healpy integration test

**Critical Regression Test**:
- `test_morton_regression_full` - Validates ALL 1,239,001 Antarctic polygon morton indices match reference
- This is the **definitive** test that Rust produces identical results to numba

**Run with**:
```bash
# With Rust (default when built)
pytest -v

# Force numba for comparison
MORTIE_FORCE_NUMBA=1 pytest -v

# Both and compare
pytest -v > rust_results.txt
MORTIE_FORCE_NUMBA=1 pytest -v > numba_results.txt
diff rust_results.txt numba_results.txt
```

### 3. Rust-vs-Numba Comparison Tests
**Purpose**: Explicit validation that Rust and numba produce identical output

**Test Cases**:
```python
import numpy as np
from mortie import tools
import os

def test_rust_vs_numba_scalar():
    """Test scalar inputs produce identical results"""
    # Force Rust
    os.environ.pop('MORTIE_FORCE_NUMBA', None)
    rust_result = tools.fastNorm2Mort(18, 1000, 2)

    # Force numba
    os.environ['MORTIE_FORCE_NUMBA'] = '1'
    numba_result = tools.fastNorm2Mort(18, 1000, 2)

    assert rust_result == numba_result

def test_rust_vs_numba_arrays():
    """Test array inputs produce identical results"""
    orders = np.array([18, 18, 18], dtype=np.int64)
    normed = np.array([100, 200, 300], dtype=np.int64)
    parents = np.array([2, 8, 5], dtype=np.int64)

    # Rust
    os.environ.pop('MORTIE_FORCE_NUMBA', None)
    rust_results = tools.fastNorm2Mort(orders, normed, parents)

    # Numba
    os.environ['MORTIE_FORCE_NUMBA'] = '1'
    numba_results = tools.fastNorm2Mort(orders, normed, parents)

    np.testing.assert_array_equal(rust_results, numba_results)

def test_rust_vs_numba_large_scale():
    """Test on Antarctic polygon data (1.2M coordinates)"""
    from pathlib import Path

    test_dir = Path(__file__).parent
    coords_file = test_dir / "Ant_Grounded_DrainageSystem_Polygons.txt"

    if not coords_file.exists():
        pytest.skip("Polygon data not available")

    data = np.loadtxt(coords_file)
    lats = data[:, 0]
    lons = data[:, 1]

    # Generate morton via geo2mort (which uses fastNorm2Mort internally)
    os.environ.pop('MORTIE_FORCE_NUMBA', None)
    rust_morton = tools.geo2mort(lats, lons, order=18)

    os.environ['MORTIE_FORCE_NUMBA'] = '1'
    numba_morton = tools.geo2mort(lats, lons, order=18)

    np.testing.assert_array_equal(rust_morton, numba_morton)
```

### 4. Performance Benchmarks
**Purpose**: Verify Rust meets or exceeds numba performance

**Rust Criterion Benchmarks**:
```bash
cargo bench
```

**Python Benchmarks**:
```python
import numpy as np
import time
from mortie import tools

def benchmark_implementation(name, force_numba=False):
    if force_numba:
        os.environ['MORTIE_FORCE_NUMBA'] = '1'
    else:
        os.environ.pop('MORTIE_FORCE_NUMBA', None)

    # Load test data
    data = np.loadtxt("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")
    lats = data[:, 0]
    lons = data[:, 1]

    # Warmup
    _ = tools.geo2mort(lats[:1000], lons[:1000], order=18)

    # Benchmark
    start = time.perf_counter()
    morton = tools.geo2mort(lats, lons, order=18)
    elapsed = time.perf_counter() - start

    print(f"{name}: {len(lats):,} coordinates in {elapsed*1000:.2f} ms ({len(lats)/elapsed:.0f} coords/sec)")
    return elapsed

# Run benchmarks
rust_time = benchmark_implementation("Rust", force_numba=False)
numba_time = benchmark_implementation("Numba", force_numba=True)
speedup = numba_time / rust_time

print(f"\nSpeedup: {speedup:.2f}x")
```

## Integration Testing Strategy

### Phase 1: Local Development
1. ✅ Build Rust extension: `maturin develop --release`
2. ✅ Run Rust unit tests: `cargo test`
3. ✅ Run Python tests with Rust: `pytest -v`
4. ✅ Run Python tests with numba: `MORTIE_FORCE_NUMBA=1 pytest -v`
5. ✅ Compare outputs: Both should produce identical results

### Phase 2: CI/CD Validation
1. ✅ GitHub Actions builds wheels for all platforms
2. ✅ Automated testing on Linux, macOS, Windows
3. ✅ Python 3.10, 3.11, 3.12 compatibility
4. ✅ Wheel installation and import testing

### Phase 3: Pre-Release Validation
1. Test wheel installation on clean environments
2. Verify fallback to numba when Rust unavailable
3. Performance benchmarking vs current release
4. Memory profiling (check for leaks)

## Success Criteria

### Correctness (MANDATORY)
- ✅ All 50 existing tests pass
- ✅ Antarctic polygon regression test passes (exact match on 1.2M+ coordinates)
- ✅ Rust unit tests all pass
- ✅ Rust and numba produce bit-identical results for all test cases

### Performance (TARGET)
- ✅ Rust ≥ numba performance on large arrays (1M+ elements)
- ✅ Rust faster on small arrays (<1000 elements) due to no JIT overhead
- ✅ Memory usage ≤ numba

### Distribution (REQUIRED)
- ✅ Wheels build for Linux (x86_64, aarch64)
- ✅ Wheels build for macOS (x86_64, aarch64)
- ✅ Wheels build for Windows (x86_64)
- ✅ Wheels install cleanly via pip
- ✅ Fallback to numba works when wheels unavailable

## Failure Scenarios & Rollback

### Scenario 1: Rust results don't match numba
**Detection**: `test_morton_regression_full` fails
**Action**: Debug morton.rs logic, verify bit-shift operations
**Rollback**: Keep numba as default, Rust opt-in via environment variable

### Scenario 2: Performance regression
**Detection**: Benchmarks show Rust slower than numba
**Action**: Profile with `cargo flamegraph`, optimize hot paths
**Rollback**: Keep numba as default

### Scenario 3: Wheel build failures
**Detection**: CI/CD build-wheels workflow fails
**Action**: Fix platform-specific issues, test locally with maturin
**Rollback**: Release without Rust, provide source-only package

### Scenario 4: Import errors on some platforms
**Detection**: test-wheels job fails
**Action**: Check PyO3/maturin compatibility, verify abi3 settings
**Rollback**: Provide platform-specific wheels only for working platforms

## Testing Checklist Before Release

- [ ] All Rust unit tests pass (`cargo test`)
- [ ] All Python tests pass with Rust (`pytest -v`)
- [ ] All Python tests pass with numba (`MORTIE_FORCE_NUMBA=1 pytest -v`)
- [ ] Explicit Rust vs numba comparison test passes
- [ ] Antarctic polygon regression test passes (1.2M coordinates)
- [ ] Benchmarks show Rust ≥ numba performance
- [ ] Memory profiling shows no leaks
- [ ] Wheels build on all platforms (CI/CD green)
- [ ] Wheels install cleanly on fresh environments
- [ ] Fallback to numba works when Rust unavailable
- [ ] Documentation updated (README, BUILDING.md)
- [ ] Version bumped in Cargo.toml and pyproject.toml

## Continuous Monitoring

### Post-Release
- Monitor PyPI download statistics
- Track GitHub issues for platform-specific bugs
- Collect performance reports from users
- Watch for Rust/PyO3/maturin updates that require code changes

## Test Data

### Reference Data Files
- `Ant_Grounded_DrainageSystem_Polygons.txt` - 1,239,001 coordinates
- `Ant_Grounded_DrainageSystem_Polygons_morton.npz` - Reference morton indices

### Test Coordinate Coverage
- ✅ Northern hemisphere (parents 0-5, positive morton indices)
- ✅ Southern hemisphere (parents 6-11, negative morton indices)
- ✅ All 12 HEALPix base cells
- ✅ Orders 6, 10, 14, 18
- ✅ Equator, poles, mid-latitudes
- ✅ All longitudes (-180 to +180)

## Known Edge Cases

1. **Parent cell boundary**: Parents 5/6 transition (sign change)
2. **Maximum order**: Order 18 at i64 limit
3. **Zero normed address**: First pixel in parent cell
4. **Maximum normed address**: Last pixel in parent cell
5. **Broadcast semantics**: Scalar order with array normed/parents

All edge cases covered by existing test suite.
