# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.3] - 2026-02-25

- fixes to PIP, updated docs, code pruning (3db1e34)
- adding multipart polygon handling (53d101c)
- fixing awful bug that expanded coverage to the full globe because of 'leaks' in the buffer 'wall' (57320eb)
- fix: adjust real-data test thresholds to match actual basin cell counts (88c51be)
- fix: use .copy() on array slices for PyO3 compatibility in closed polygon handling (5bddaba)
- first pass rust implementation (0f3bb10)
- docs: update CHANGELOG.md for 0.6.2 (22ab907)

## [0.6.2] - 2026-02-25

- Add step parameter to boundaries() and mort2polygon() ([#19](https://github.com/espg/mortie/pull/19)) by @espg


## [0.6.1] - 2026-02-25

- Fix wheel test import and run on all pushes ([#18](https://github.com/espg/mortie/pull/18)) by @espg


## [0.6.0] - 2026-02-25

- major feature: morton_buffer for spatial cell expansion ([#17](https://github.com/espg/mortie/pull/17)) by @espg
- Set up CodSpeed for continuous performance monitoring ([#16](https://github.com/espg/mortie/pull/16)) by @codspeed-hq
- major refactor: Rust-native HEALPix, no Python backends ([#15](https://github.com/espg/mortie/pull/15)) by @espg
- Robust spanning tree algorithm to replace greedy_polygon ([#14](https://github.com/espg/mortie/pull/14)) by @espg


## [0.5.2] - 2025-12-10

- update for numpy 2 compat ([#13](https://github.com/espg/mortie/pull/13)) by @espg


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
