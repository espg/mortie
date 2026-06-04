# Polygon coverage methods

`mortie` covers a polygon with HEALPix cells using a **top-down hierarchical
region coverer**: starting from the 12 base cells it keeps cells that are inside
the polygon, prunes cells that are outside, and refines cells the boundary
passes through — down to a target order. Cost scales with the *boundary*, not
the polygon's area or vertex count, so it stays fast on very large, high-vertex
polygons.

Two output shapes and two adaptive stop criteria are available.

## Output shapes

| function | output | when to use |
|---|---|---|
| `morton_coverage(lats, lons, order)` | **flat** — every cell at `order` | you need a uniform-resolution cell list |
| `morton_coverage_moc(lats, lons, order)` | **MOC** — mixed order (coarse interior, fine boundary) | you want a compact, exact cover; usually far smaller |

Both are exact (contract: a cell is included iff it intersects the closed
polygon — the cover is a guaranteed superset of the polygon). Because a mortie
morton index self-encodes its order, the MOC is still a plain `int64` array.

## Adaptive stop criteria (`morton_coverage_moc` only)

Mutually exclusive; both trade boundary precision for fewer cells and less time:

- `tolerance=<degrees>` — stop refining a boundary cell once its angular radius
  drops below `tolerance`. The boundary precision is fixed in **angular** terms
  and is independent of `order`.
- `max_cells=<n>` — refine the largest boundary cells first until about `n`
  cells, giving an **adaptive** boundary: fine where it wiggles, coarse where it
  is straight. If `n` is below the minimum needed to represent the polygon it is
  raised to that floor and a warning is emitted.

Both criteria are **order-independent in effect**: `tolerance` fixes boundary
precision in angular terms, and `max_cells` stops before reaching the finest
order, so raising `order` past where either kicks in does not change the result.
(That is why the `tol 0.5°` column below is identical across orders 8/10/12.)

All methods are deterministic (a pure function of the inputs).

## Multipart polygons and holes

Pass `lats`/`lons` as a **list of rings** to cover a multipart polygon or a
polygon with holes. All rings are covered by one even-odd descent — a cell is
covered iff its centre is inside an *odd* number of rings — which means:

- **Disjoint parts** union (with no seam along a shared interior border).
- A **nested ring carves a hole**: a donut is `[outer, hole]`; nesting depth
  decides inside/outside, so ring orientation does not matter.

```python
# Donut: an outer box with a rectangular hole
outer_lat, outer_lon = [35, 35, 55, 55], [-130, -110, -110, -130]
hole_lat,  hole_lon  = [42, 42, 48, 48], [-123, -117, -117, -123]
donut = mortie.morton_coverage([outer_lat, hole_lat], [outer_lon, hole_lon], order=8)

# Multipart: two disjoint triangles, unioned
multi = mortie.morton_coverage([latsA, latsB], [lonsA, lonsB], order=8)
```

`morton_coverage_moc` accepts the same list-of-rings form (the per-part MOCs are
unioned and compressed).

> Note: the coverer does not *dissolve* shared borders. If you cover a set of
> polygons that tile a region (e.g. drainage basins), the cells along their
> shared borders are — correctly — boundary cells. To cover the dissolved
> outline as one region, union the polygons geometrically first.

## MOC helpers

- `compress_moc(morton)` — collapse a morton set to its canonical compact MOC
  (merge any 4 complete sibling cells into their parent; drop any cell contained
  in a coarser one). Use after unioning covers from several polygons.
- `moc_to_order(morton, order)` — densify a mixed-order MOC back to a flat list
  at `order`. `moc_to_order(morton_coverage_moc(...), order)` reproduces exactly
  `morton_coverage(..., order)` — the MOC is a lossless, compact encoding of the
  same cover.

## Benchmark matrix

Canonical Antarctic drainage basin (82k vertices, and densified to 1M), median
wall-clock; `old (flat)` is the previous flood-fill implementation.

| verts | order | old (flat) | hier flat | hier MOC | MOC + tol 0.5° | MOC + budget 2k |
|--:|--:|--|--|--|--|--|
| 81,595 | 8 | 883c / 151ms | 883c / 65ms | 196c / 43ms | 79c / 53ms | 196c / 58ms |
| 81,595 | 10 | 12,459c / 2,989ms | 12,461c / 60ms | 1,058c / 58ms | 79c / 44ms | 867c / 62ms |
| 81,595 | 12 | (too slow) | 191,710c / 74ms | 5,146c / 65ms | 79c / 40ms | 867c / 63ms |
| 1,000,000 | 10 | 12,461c / 45,755ms | 12,461c / 1,190ms | 1,058c / 1,079ms | 79c / 1,025ms | 867c / 1,489ms |

`c` = cell count, `ms` = milliseconds. Reproduce with `bench_matrix.py`.

### Reading the matrix

- **Hierarchical vs. old:** at working resolution the new coverer is ~40–60×
  faster (82k @ order 10: 2,989 ms → 60 ms; 1M @ order 10: 45.8 s → 1.2 s).
- **MOC vs. flat:** identical coverage, far fewer cells — at order 12 the flat
  cover is 191,710 cells but the MOC is 5,146 (≈37× smaller) at the same speed.
  Prefer the MOC unless you specifically need uniform-order leaves.
- **Adaptive criteria:** `tolerance` and `max_cells` cut cells and time further
  when an approximate boundary is acceptable. Note the `tolerance=0.5°` row is
  identical (79 cells) across orders 8/10/12 — angular precision is fixed
  regardless of the order ceiling.
- **Very large vertex counts:** at 1M vertices the runtime is dominated by a
  one-time O(V) setup (building edges + the 12 base-cell tests); the descent
  itself is nearly free, so higher orders cost little extra.
