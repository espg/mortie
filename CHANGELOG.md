# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- **Segmented toc reduce: `tocs_reduce`** (issue #177)
  ([#192](https://github.com/espg/mortie/pull/192)). The ragged sibling of
  `toc_reduce`: ragged `(words, offsets)` in arrow list layout in, one merged
  `uint64` out per group, GIL released and rayon across groups. Result `i` is
  bit-identical to `toc_reduce` on group `i` alone — same semilattice join,
  same instant preservation, same fold-tree independence. An **empty group is
  an error**, not an empty slot (the merge has no identity element), and
  layout failures name the offending group. The consumer is a per-cell
  temporal fold (englacial/zagg#410). The interval-set algebra of issue #177
  (`normalize` / union / intersect / minus) remains deferred.

## [0.9.8] - 2026-08-16

- Authalic latitude convention: new default latitude="authalic" (issue #186) ([#188](https://github.com/espg/mortie/pull/188)) by @espg
- Winding-free dissolve classifier: hemisphere+ covers dissolve instead of raising (issue #147) ([#182](https://github.com/espg/mortie/pull/182)) by @espg


- **BREAKING: latitude convention — authalic on WGS84 by default** (issue #186)
  ([#188](https://github.com/espg/mortie/pull/188)). Every geodetic lat/lon
  crossing now takes a keyword-only `latitude=` parameter. The new default
  `"authalic"` converts WGS84 geodetic latitude to authalic latitude before
  the spherical HEALPix kernel (and back on output), making cells
  **equal-area on the ellipsoid by construction**. The pre-change behavior
  is available on every surface as `latitude="geodetic-spherical"`.
  **Cell ids under the two conventions are non-corresponding**: the same
  coordinates hash to different morton words (identical at the equator and
  poles, drifting to ~0.128 deg / ~14.26 km of latitude at 45 deg), so
  pinned cell ids computed with earlier versions reproduce only under the
  legacy escape. Conversion accuracy is <= 1e-13 rad per direction
  (docs/specification.md §9). New helpers `geodetic_to_authalic` /
  `authalic_to_geodetic` expose the raw mapping.

## [0.9.7] - 2026-08-09

- Deterministic, order-independent ring chaining in dissolve (issue #155) ([#179](https://github.com/espg/mortie/pull/179)) by @espg


## [0.9.6] - 2026-08-09

- toc word: temporal order coverage (issue #175) ([#178](https://github.com/espg/mortie/pull/178)) by @espg
- Batch MOC set ops: mocs_and / mocs_intersect + scalar moc_intersects (issue #173) ([#174](https://github.com/espg/mortie/pull/174)) by @espg
- Resurrect mortie/batch.py as the consolidated home for the bulk operators ([#172](https://github.com/espg/mortie/pull/172)) by @espg
- Split tools.py into convert/orders/buffer (issue #159) ([#169](https://github.com/espg/mortie/pull/169)) by @espg


## [0.9.5] - 2026-08-08

- mortie.arrow.from_wkbs: the pyarrow skin over the WKB batch (issue #163) ([#167](https://github.com/espg/mortie/pull/167)) by @espg
- Rust WKB reader: backend-free geometry ingest, plus from_wkbs (issue #157) ([#158](https://github.com/espg/mortie/pull/158)) by @espg
- common_ancestors + children_of: the dense-output batch pair (issue #156 phase 3) ([#164](https://github.com/espg/mortie/pull/164)) by @espg
- Batch MOC densify: mocs_to_orders + the mortie/moc.py extraction (issue #156) ([#160](https://github.com/espg/mortie/pull/160)) by @espg


## [0.9.4] - 2026-08-07

- Batch polygon coverage: polygons_to_morton_mocs (issue #153) ([#154](https://github.com/espg/mortie/pull/154)) by @espg
- rank_to_xy / xy_to_rank: rank-space (x, y) deinterleave (issue #149) ([#150](https://github.com/espg/mortie/pull/150)) by @espg


## [0.9.3] - 2026-07-29

- Group A small-fix bundle from issue #108 ([#111](https://github.com/espg/mortie/pull/111)) by @espg


## [0.9.2] - 2026-07-26

- Ring validity: bucketed simplicity check, oracle fixtures, and the cap-axis fast path ([#146](https://github.com/espg/mortie/pull/146)) by @espg
- Repair the antipodal-lens defect in the point-in-ring reference layer (phase 1 of #107) ([#138](https://github.com/espg/mortie/pull/138)) by @espg
- convert tools / coverage / prefix_trie / linestring docstrings to numpydoc ([#143](https://github.com/espg/mortie/pull/143)) by @espg
- Convert morton_index, geometry and arrow docstrings to numpydoc ([#141](https://github.com/espg/mortie/pull/141)) by @espg
- UNIQ helpers: multi-resolution support, and drop uniq2geo's untrusted order ([#139](https://github.com/espg/mortie/pull/139)) by @espg
- Freeze audit: refresh stale ≤18-era / signed-decimal docstrings (issue #68) ([#130](https://github.com/espg/mortie/pull/130)) by @espg
- Docs site (mkdocs-material + mkdocstrings) and build workflow (issue #133) ([#137](https://github.com/espg/mortie/pull/137)) by @espg


## [0.9.1] - 2026-07-25

- Public decimal→word parse API (issue #114) ([#132](https://github.com/espg/mortie/pull/132)) by @espg
- Cause-tagged node_straddles instrumentation + over-refinement measurement (issue #90, phases 1-2) ([#112](https://github.com/espg/mortie/pull/112)) by @espg
- refresh README benchmarks + cross-order table (issue #65) ([#131](https://github.com/espg/mortie/pull/131)) by @espg
- add HEALPix interchange guide (issue #63) ([#128](https://github.com/espg/mortie/pull/128)) by @espg
- add morton_index datatype user docs (issue #64) ([#129](https://github.com/espg/mortie/pull/129)) by @espg
- Mixed-order support in the geo kernels (issue #116) ([#122](https://github.com/espg/mortie/pull/122)) by @espg
- freeze path_grouping remainder rule in spec §6.1 ([#127](https://github.com/espg/mortie/pull/127)) by @espg
- Unify order2res Earth model with the spec-page sphere (R=6371.0088) ([#126](https://github.com/espg/mortie/pull/126)) by @espg
- fix test_spec_page order-29 tie-break probe (issue #123) ([#125](https://github.com/espg/mortie/pull/125)) by @espg
- Uniform symbolic crossing predicate: fix base-cell-boundary mis-fill ([#106](https://github.com/espg/mortie/pull/106)) by @espg
- Decimal kind suffix p for point ids (issue #120) ([#121](https://github.com/espg/mortie/pull/121)) by @espg
- v1.0 specification & conventions page (issue #62) ([#118](https://github.com/espg/mortie/pull/118)) by @espg
- Gate the seed cull to sub-hemisphere caps ([#110](https://github.com/espg/mortie/pull/110)) by @espg
- Recover coverage perf after the #103 predicate swap ([#109](https://github.com/espg/mortie/pull/109)) by @espg


## [0.9.0] - 2026-07-09

- Fix res2display() order cap and add km/m/cm unit ladder ([#102](https://github.com/espg/mortie/pull/102)) by @espg
- Decimal-string display & casting layer for morton_index ([#105](https://github.com/espg/mortie/pull/105)) by @espg
- numpy-level point-kind geo2mort: geo2mort(..., points=True) ([#100](https://github.com/espg/mortie/pull/100)) by @espg


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


## [0.8.2] - 2026-06-25

- moc xor + domain-bounded not ([#59](https://github.com/espg/mortie/pull/59)) by @espg
- Lift coverage/MOC/set-op order cap from 18 to 29 ([#70](https://github.com/espg/mortie/pull/70)) by @espg


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
