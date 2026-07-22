# Polygon coverage methods

`mortie` covers a polygon with HEALPix cells using a **top-down hierarchical
region coverer**: starting from the 12 base cells it keeps cells that are inside
the polygon, prunes cells that are outside, and refines cells the boundary
passes through — down to a target order. Cost scales with the *boundary*, not
the polygon's **area** (interior regions collapse to a few coarse cells), so a
large but simple polygon is cheap. Vertex count still matters — there is a
one-time `O(V)` setup and per-boundary-cell work grows with local edge density —
but far more gently than the old `O(cells × vertices)` flood-fill (a 1M-vertex
polygon covers ~40× faster); see the benchmark matrix below.

> **First-call warm-up.** The first coverage call in a process spins up the
> `rayon` threadpool and runs on cold caches — a one-time cost that is a large
> fraction of the runtime for a *small* cover (several times the warm time),
> though negligible for a large one. If first-call latency matters (a request path or
> interactive tool), warm it once at startup with a throwaway call —
> e.g. `morton_coverage_moc(box_lats, box_lons, order=6)` — before the calls you
> care about. The *First-call warm-up cost* table under the benchmark matrix
> below measures it on a real MOC cover; steady-state timings are what the matrix
> and [benchmarks.md](benchmarks.md) report.

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
  decides inside/outside.

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

### Ring winding (orientation)

mortie follows the [RFC 7946 §3.1.6](https://datatracker.ietf.org/doc/html/rfc7946#section-3.1.6)
/ S2 **right-hand rule** for ring orientation:

- **Exterior rings** are wound **counter-clockwise** (CCW) — the interior is on
  the **left** of each directed edge.
- **Holes** are wound **clockwise** (CW).

For a ring whose **vertices fit inside a hemisphere** the two regions split into
an unambiguous smaller and larger side, and the smaller side is the interior. So
that everyday clockwise input doesn't silently invert, mortie **normalizes the
orientation of such rings at ingest** — it reads each ring's winding sign once
(an O(V) check) and reverses a clockwise ring to CCW. The practical upshot:
**for ordinary, sub-hemisphere polygons you may pass rings in either winding and
get the same cover** (this matches the usual GIS "smaller-area-is-interior"
behaviour).

**Orientation becomes load-bearing for hemisphere-plus polygons** — e.g. "the
whole globe except Antarctica". On a sphere a closed ring splits the surface into
two regions of equal standing, so the vertex set alone cannot say which is
"inside"; only the winding direction disambiguates. The robust backend (issue
#22) keys on that direction, so mortie **never reorders a ring whose vertices
span a hemisphere or more** — those are trusted exactly as supplied, and a ring
wound the wrong way deliberately selects the *complementary* region.

Note one consequence: a region whose *interior* exceeds a hemisphere but whose
*boundary vertices* still sit within one (the Antarctica-hugging ring of "all but
Antarctica" lies in a sub-hemisphere cap) would be normalized back to its small
side. Express such a region the way GeoJSON authors it anyway — a whole-world
outer ring with a small hole, or vertices that genuinely span the sphere — rather
than as a lone sub-hemisphere-vertex ring relying on reversed winding. When in
doubt, wind exteriors CCW and holes CW.

## MOC helpers

- `compress_moc(morton)` — collapse a morton set to its canonical compact MOC
  (merge any 4 complete sibling cells into their parent; drop any cell contained
  in a coarser one). Use after unioning covers from several polygons.
- `moc_to_order(morton, order)` — densify a mixed-order MOC back to a flat list
  at `order`. `moc_to_order(morton_coverage_moc(...), order)` reproduces exactly
  `morton_coverage(..., order)` — the MOC is a lossless, compact encoding of the
  same cover.

## Benchmark matrix

Canonical Antarctic drainage basin (full ~81.6k vertices, and densified to 1M).
The previous flood-fill implementation took **2,989 ms** for this basin at order
10 and **45.8 s** at 1M vertices; the hierarchical coverer below is **~40–60×**
faster at working resolution.

The table below is regenerated by `bench_matrix.py` (run from the repo root) —
it writes itself in place between the markers:

<!-- BENCH_MATRIX:START -->

| verts | order | flat | MOC | MOC tol 0.5° | MOC tol 0.05° | MOC budget 2k | MOC budget 500 |
|--:|--:|--|--|--|--|--|--|
| 81,595 | 8 | 883c / 148ms | 196c / 148ms | 79c / 132ms | 196c / 136ms | 196c / 182ms | 196c / 178ms |
| 81,595 | 10 | 12,461c / 168ms | 1,058c / 162ms | 79c / 139ms | 1,058c / 156ms | 867c / 200ms | 200c / 179ms |
| 81,595 | 12 | 191,710c / 195ms | 5,146c / 199ms | 79c / 137ms | 2,039c / 172ms | 867c / 188ms | 200c / 175ms |
| 1,000,000 | 10 | 12,461c / 2893ms | 1,058c / 2820ms | 79c / 2302ms | 1,058c / 2759ms | 867c / 3208ms | 200c / 2997ms |

`c` = cell count, `ms` = milliseconds. Matrix timings are the warm median (each method is called once to warm up, then timed); see the first-call warm-up table for the one-time cold cost. Timings are machine/run dependent; cell counts are deterministic.

<!-- BENCH_MATRIX:END -->

### First-call warm-up cost

The matrix timings above are the **warm** (steady-state) median. The **first**
`morton_coverage_moc` call in a process additionally pays a one-time cost (the
`rayon` threadpool spins up, caches are cold). Because that cost is fixed, its
weight is inversely proportional to the cover's size — it dominates a tiny cover
and vanishes on a large one. Measured as a genuine first call (each row in its
own fresh process):

<!-- BENCH_WARMUP:START -->

| MOC cover | cold (first call) | warm (steady state) | ratio |
|--|--:|--:|--:|
| ~1 km box, order 11 | 1.1 ms | 0.2 ms | 6.2x |
| 81.6k-vert basin, order 10 | 178.6 ms | 159.7 ms | 1.1x |

<!-- BENCH_WARMUP:END -->

So a realistic basin cover barely notices it, but a small cover runs several
times slower on its first call — warm once at startup (above) if that matters.

### Reading the matrix

- **MOC vs. flat:** identical coverage, far fewer cells — at order 12 the flat
  cover is ~192k cells but the MOC is ~5k (≈37× smaller) at the same speed.
  Prefer the MOC unless you specifically need uniform-order leaves.
- **Adaptive criteria:** `tolerance` and `max_cells` cut cells and time further
  when an approximate boundary is acceptable. Note the `tolerance=0.5°` row is
  identical (79 cells) across orders 8/10/12 — angular precision is fixed
  regardless of the order ceiling.
- **Very large vertex counts:** at 1M vertices the runtime is dominated by a
  one-time O(V) setup (building edges + the 12 base-cell tests); the descent
  itself is nearly free, so higher orders cost little extra.
