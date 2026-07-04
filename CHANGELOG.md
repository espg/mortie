# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.5] - 2026-07-04

- Wire up the arro3-no-pyarrow CI leg ([#101](https://github.com/espg/mortie/pull/101)) by @espg


- numpy-level point-kind `geo2mort(..., points=True)` encoder; lat/lon now default to order-29 point words (issue #96). **Breaking:** a bare `geo2mort(lat, lon)` returns order-29 `Kind::Point` words (was order-18 area cells) — pass an explicit `order` for an area cell. Non-finite lat/lon encode to the reserved `0`. ([#100](https://github.com/espg/mortie/pull/100)) by @espg

## [0.8.4] - 2026-07-01

- Harden release changelog/version commit against non-tip tags ([#95](https://github.com/espg/mortie/pull/95)) by @espg
- Library-agnostic Arrow C Data Interface for morton_index (arro3-core / PyCapsule) ([#94](https://github.com/espg/mortie/pull/94)) by @espg


## [0.8.3] - 2026-06-30

- Tag-driven Cargo.toml version sync ([#91](https://github.com/espg/mortie/pull/91)) by @espg
- WKB/WKT geometry I/O: ingest + dissolved emit (issue #71) ([#89](https://github.com/espg/mortie/pull/89)) by @espg
- morton index surface followup: points= encode + __from_arrow__ ([#86](https://github.com/espg/mortie/pull/86)) by @espg
- SoS-harden on_minor_arc in robust_crossing ([#87](https://github.com/espg/mortie/pull/87)) by @espg
- small fixes 2026-06-27: densify size guard (#80) + morton_polygon determinism tests (#83) ([#85](https://github.com/espg/mortie/pull/85)) by @espg
- Update example notebooks for order-29 packed encoding + binder wheel ([#76](https://github.com/espg/mortie/pull/76)) by @espg
- add split_base_cells ([#84](https://github.com/espg/mortie/pull/84)) by @espg
- moc_min / common_ancestor: deepest-common-ancestor reduction ([#72](https://github.com/espg/mortie/pull/72)) by @espg


## [0.8.3] - 2026-06-30

- WKB/WKT geometry I/O: ingest + dissolved emit (issue #71) ([#89](https://github.com/espg/mortie/pull/89)) by @espg
- morton index surface followup: points= encode + __from_arrow__ ([#86](https://github.com/espg/mortie/pull/86)) by @espg
- SoS-harden on_minor_arc in robust_crossing ([#87](https://github.com/espg/mortie/pull/87)) by @espg
- small fixes 2026-06-27: densify size guard (#80) + morton_polygon determinism tests (#83) ([#85](https://github.com/espg/mortie/pull/85)) by @espg
- Update example notebooks for order-29 packed encoding + binder wheel ([#76](https://github.com/espg/mortie/pull/76)) by @espg
- add split_base_cells ([#84](https://github.com/espg/mortie/pull/84)) by @espg
- moc_min / common_ancestor: deepest-common-ancestor reduction ([#72](https://github.com/espg/mortie/pull/72)) by @espg


## [0.8.2] - 2026-06-25

- moc xor + domain-bounded not ([#59](https://github.com/espg/mortie/pull/59)) by @espg
- Lift coverage/MOC/set-op order cap from 18 to 29 ([#70](https://github.com/espg/mortie/pull/70)) by @espg


## [0.8.2] - 2026-06-25

- Lift coverage/MOC/set-op order cap from 18 to 29 ([#70](https://github.com/espg/mortie/pull/70)) by @espg


## [0.8.1] - 2026-06-19

- packed-u64 migration (Option A) + norm2mort consolidation ([#58](https://github.com/espg/mortie/pull/58)) by @espg
- #34 §D cleanup + fmt/clippy sweep ([#57](https://github.com/espg/mortie/pull/57)) by @espg
- remove MORTIE_FORCE_PYTHON parity fallbacks ([#49](https://github.com/espg/mortie/pull/49)) by @espg


## [0.8.1] - 2026-06-19

- packed-u64 migration (Option A) + norm2mort consolidation ([#58](https://github.com/espg/mortie/pull/58)) by @espg
- #34 §D cleanup + fmt/clippy sweep ([#57](https://github.com/espg/mortie/pull/57)) by @espg
- remove MORTIE_FORCE_PYTHON parity fallbacks ([#49](https://github.com/espg/mortie/pull/49)) by @espg


## [0.8.0] - 2026-06-18

- Update Cargo.toml for 0.8.0 release ([#55](https://github.com/espg/mortie/pull/55)) by @espg
- MOC boolean set algebra via the patched healpix BMOC fork ([#53](https://github.com/espg/mortie/pull/53)) by @espg
- morton_index datatype skin: pandas + Arrow (phases 4 & 5 of #35) ([#51](https://github.com/espg/mortie/pull/51)) by @espg
- Robust hemisphere+ point-in-polygon (S2-style orientation + SoS) ([#44](https://github.com/espg/mortie/pull/44)) by @espg
- decimal_morton: full-resolution 64-bit Morton MOC kernel (phase 1) ([#43](https://github.com/espg/mortie/pull/43)) by @espg
- Update CLAUDE.md ([#46](https://github.com/espg/mortie/pull/46)) by @espg
- Update CLAUDE.md ([#45](https://github.com/espg/mortie/pull/45)) by @espg
- #34 perf cluster (before #35): GIL release + descent hot-loop + micro-wins + batched vec2ang ([#41](https://github.com/espg/mortie/pull/41)) by @espg
- CLAUDE.md: per-issue claude/ branches, multi-PR, and PR label states ([#42](https://github.com/espg/mortie/pull/42)) by @espg
- small fixes: relicense to MIT, ruff lint workflow ([#40](https://github.com/espg/mortie/pull/40)) by @espg
- Variable cell densification   ([#33](https://github.com/espg/mortie/pull/33)) by @espg


## [0.7.2] - 2026-06-06

- Variable cell densification   ([#33](https://github.com/espg/mortie/pull/33)) by @espg


## [0.7.1] - 2026-06-03

- feat: hierarchical coverage from polygon (correctness, native MOC coverage, 'donut' polygon support) ([#31](https://github.com/espg/mortie/pull/31)) by @espg


## [0.7.0] - 2026-06-03

- coverage bug fix ([#29](https://github.com/espg/mortie/pull/29)) by @espg


## [0.6.6] - 2026-06-03

- coverage bug fix (453f767)
- executed notebooks with outputs (30c41dc)
- notebook updates, removing healpy (fac8165)
- docs: update CHANGELOG.md for 0.6.5 (b6628fc)

## [0.6.5] - 2026-04-15

- Linestring / multi-linestring morton coverage + metric buffer helper ([#26](https://github.com/espg/mortie/pull/26)) by @espg


## [0.6.4] - 2026-03-10

- Add polygon-to-morton coverage function ([#21](https://github.com/espg/mortie/pull/21)) by @espg


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
