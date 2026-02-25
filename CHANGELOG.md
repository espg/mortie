# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Renamed `morton_bbox` module to `prefix_trie` (Python, Rust, and tests)
- Renamed `refine_bbox()` → `morton_polygon()`, `refine_bbox_geo()` → `geo_morton_polygon()`, `refine_bbox_morton()` → `morton_polygon_from_array()`
- HEALPix backend is now abstracted via `mortie._healpix`; supports both `healpy` and `cdshealpix`
- `healpy` is no longer a hard dependency; install via `pip install mortie[healpy]` or `pip install mortie[cdshealpix]`
- Moved pytest config from `pytest.ini` to `pyproject.toml`

### Added
- `mortie/_healpix.py` backend abstraction layer with auto-detection
- Cross-backend comparison tests (`test_healpix_backends.py`)
- CodSpeed performance benchmarks (`benchmarks/test_bench_cpu.py`)
- CodSpeed CI workflow (`.github/workflows/codspeed.yml`)
- `examples/` and `benchmarks/` directories for better organization

### Removed
- `morton_bounding_box()` — no-op wrapper, use `split_children()` directly
- `setup.cfg` and `setup.py` — superseded by `pyproject.toml` + maturin
- `pytest.ini` — config moved to `pyproject.toml`
- Unused `pandas` and `cython` from `environment.yml`

## [0.5.2] - 2025-12-10

- update for numpy 2 compat ([#13](https://github.com/espg/mortie/pull/13)) by @espg
- Efficient conversion of complex polygons to morton coverage ([#12](https://github.com/espg/mortie/pull/12)) by @espg


## [0.5.1] - 2025-11-25

- Efficient conversion of complex polygons to morton coverage ([#12](https://github.com/espg/mortie/pull/12)) by @espg


## [0.5.0] - 2025-11-19

- fixing test error (c4ef6b5)
- docs: update CHANGELOG.md for 0.4.10 (4ce1862)

## [0.4.10] - 2025-11-19

- Fix antimeridian normalization in mort2polygon and mort2bbox ([#10](https://github.com/espg/mortie/pull/10)) by @espg


## [0.4.8] - 2025-11-19

- Fix antimeridian normalization in mort2polygon and mort2bbox ([#10](https://github.com/espg/mortie/pull/10)) by @espg


## [0.4.7] - 2025-11-19

- HOTFIX: Fix geo2mort function signature bug ([#9](https://github.com/espg/mortie/pull/9)) by @espg


## [0.4.6] - 2025-11-13

- HOTFIX: Fix geo2mort function signature bug ([#9](https://github.com/espg/mortie/pull/9)) by @espg


## [0.4.5] - 2025-11-13

- Fix package namespace: Include mortie Python package in wheel ([#7](https://github.com/espg/mortie/pull/7)) by @espg


## [0.4.4] - 2025-11-13

- Update macOS runners from deprecated macos-13 to macos-15-intel and macos-latest (3e88a00)
- docs: update CHANGELOG.md for 0.4.3 (2a340e5)

## [0.4.3] - 2025-11-13

- Fix Windows build by forcing bash shell for version update (8a5e872)
- docs: update CHANGELOG.md for 0.4.2 (a1639c7)

## [0.4.2] - 2025-11-13

- Fix build-wheels workflow to run on branch pushes (b80e1af)
- docs: update CHANGELOG.md for 0.4.1 (43f86af)

## [0.4.1] - 2025-11-13

- Rust <> Python API compatibility ([#6](https://github.com/espg/mortie/pull/6)) by @espg


## [0.4.0] - 2025-11-13

- New Feature/rust acceleration ([#5](https://github.com/espg/mortie/pull/5)) by @espg
- Feature/unit tests and ci ([#4](https://github.com/espg/mortie/pull/4)) by @espg
