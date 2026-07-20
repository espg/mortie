# mortie specification & conventions (v1.0)

This page is the **normative record** of the mortie 1.x stability contract:
the packed-word encoding, the decimal string grammar, the morton-hive store
layout, the coverage-MOC serializations, and the zarr DGGS convention block
for morton-declared stores. Everything marked **contract** here is frozen for
the 1.x major-version series; anything not marked contract is informative.

Design *rationale* — why each decision was made, with trade studies and
ratification records — lives in zagg's
[`docs/design/sparse_coverage.md`](https://github.com/englacial/zagg/blob/main/docs/design/sparse_coverage.md)
(the decisions registry, D-numbered entries). That document *cites* this page;
this page is the spec. Grammars and constants are normative **here only** —
duplicated normative text drifts.

Contents:

1. [The packed 64-bit morton word](#1-the-packed-64-bit-morton-word)
2. [Decimal string representation](#2-decimal-string-representation)
3. [Resolution table](#3-resolution-table)
4. [Order-29 point encodings: `resolution` discriminator and the 29→24 clip rule](#4-order-29-point-encodings)
5. [Zarr DGGS convention block](#5-zarr-dggs-convention-block)
6. [Morton-hive store layout](#6-morton-hive-store-layout)
7. [Coverage MOC serializations](#7-coverage-moc-serializations)
8. [Frozen for 1.x](#8-frozen-for-1x)

---

## 1. The packed 64-bit morton word

**Contract.** A morton index is one unsigned 64-bit word encoding a HEALPix
NESTED cell (or an order-29 point) with its order carried intrinsically.
Source of truth in code: `src_rust/src/decimal_morton.rs` (`MAX_ORDER = 29`,
`BODY_TUPLES = 27`).

### Bit layout (MSB → LSB)

```text
[ 4-bit prefix ][ 54-bit body (27 x 2-bit) ][ 6-bit suffix ]
  63 .. 60        59 ..  6                     5 ..  0
```

- **prefix** (bits 63–60) — the HEALPix base cell stored as `base_id + 1`,
  so the 12 base cells occupy `1..=12`. `0` is the empty/null sentinel;
  `13..=15` are invalid. The `+1` shift is monotonic, so a raw unsigned sort
  is preserved as a Z-order curve.
- **body** (bits 59–6) — 27 two-bit tuples, one per order `1..=27`. Order 1
  occupies the highest tuple (bits 59–58), order 27 the lowest (bits 7–6).
  The stored value is `0..=3`, *interpreted* as `1..=4` (a decode-time `+1`,
  matching the decimal grammar in §2).
- **suffix** (bits 5–0) — one plain unsigned integer `0..=63`, a preorder
  numbering of the path tail past tuple 27:

  | suffix | meaning |
  |---|---|
  | `0..=27` | variable-length **area** element; the order *is* the suffix value (`0` = base-cell-only). `27` = order-27 with empty tail. |
  | `28..=47` | order-28/29 **area** cells in parent-first preorder: `r = t28*5 + (t29_present ? t29 + 1 : 0)`, `suffix = 28 + r`, with `t28`/`t29` the **stored** `0..=3` tuple values. Each `t28` owns a 5-block: `[t28]` then its four `[t28,t29]` children. |
  | `48..=63` | order-29 **point**, max-encoded (no area claim): `r2 = t28*4 + t29`, `suffix = 48 + r2`. A point sorts *after* every area cell sharing its body (highest suffix range) — the finest, last thing there. |

- **Canonical zero-fill**: every bit below an element's order is zero-filled,
  so two encodings of the same cell are bit-equal — integer equality,
  hashing, dedup, and the raw sort all work on the word directly.
- **Unsigned storage**: the word is stored and exchanged as `uint64`. The
  signed "negative = southern" form is a *presentation* detail of the decimal
  string (§2), never a storage form. Reinterpreting the word as `int64` is an
  error: base cells 7–11 set bit 63 and would read back negative.
- **Z-order**: a raw unsigned sort of packed words is a Z-order (Morton)
  curve traversal, parent-before-children across the whole order range
  0–29.

### Kind: area vs point

A decoded word is either an **area** element (a real cell with spatial
extent, orders 0–29) or an order-29 **point** (a location cast to maximum
resolution with *no area claim* — e.g. a raw lat/lon conversion). The kind is
carried by the suffix range, not by any external flag. See §4 for the
declared-metadata consequences.

## 2. Decimal string representation

**Contract.** The decimal string is the **render-only** external form of a
morton index (paths, logs, inventories, display). Packed `uint64` words are
the storage and compute form; output types are never data-dependent:

- **strings** for display/interchange at every order;
- **packed `uint64`** for storage and compute;
- legacy signed `i64` only via the explicit, capped `to_legacy_i64()` escape
  hatch.

Grammar:

```text
morton-decimal = ["-"] base-digit *order-digit
base-digit     = "1" / "2" / "3" / "4" / "5" / "6"
order-digit    = "1" / "2" / "3" / "4"
```

- **Sign + base digit** form a constant-width component with 12 values
  (`1..6` / `-1..-6`; the sign renders the southern base cells). One
  order-digit follows per order; digits are `1-4`, never `0`. The string
  length minus the sign/base component *is* the order.
- **String prefix = spatial ancestor** at every level: `-31123` is a
  descendant of `-311`, and lexicographic grouping under a fixed-width
  prefix is spatial grouping.
- The **base component grammar** is `-?[1-6]` — reserved wherever names must
  be distinguishable from morton components (see §6.5).

### Order-29 parse non-injectivity

An order-29 **point** renders identically to the order-29 **area** cell on
the same path — kind is *not* part of the decimal repr. Parsing a decimal
string back therefore always yields the **area** word. **Never round-trip
point-ness through the string**; kind survives only in the packed word (§1)
or in declared metadata (§4).

## 3. Resolution table

Informative (the formulas are contract; the rendered values are derived).
`nside = 2^order`; the cell-scale column is `order2res(order)` =
`111 × 58.6323 × 0.5^order` km (`mortie.tools.order2res`); the area column is
the exact HEALPix cell area `4πR² / (12 · 4^order)` with `R = 6371.0088` km.
The table below is generated from `order2res` and pinned by
`mortie/tests/test_spec_page.py` so it cannot drift.

**Note — the two columns use different Earth models.** Cell scale keeps the
historical `order2res` constant (`111 km/deg × 58.6323`, an implied sphere of
`R ≈ 6366 km`); cell area uses the exact HEALPix sphere area at mean radius
`R = 6371.0088 km`. So `sqrt(area)` and cell scale diverge by ~0.2%. Both are
informative, and the drift pin only proves each column is internally
consistent with its own constant. Whether to unify the two radii is an open
question (see this PR's discussion); until it is resolved the constants stand
as documented here.

<!-- table:order2res:begin -->
| order | nside | cell scale | cell area |
|---|---|---|---|
| 0 | 1 | 6,508.185 km | 4.25055e+07 km2 |
| 1 | 2 | 3,254.093 km | 1.06264e+07 km2 |
| 2 | 4 | 1,627.046 km | 2.65659e+06 km2 |
| 3 | 8 | 813.523 km | 664,148 km2 |
| 4 | 16 | 406.762 km | 166,037 km2 |
| 5 | 32 | 203.381 km | 41,509.3 km2 |
| 6 | 64 | 101.690 km | 10,377.3 km2 |
| 7 | 128 | 50.845 km | 2,594.33 km2 |
| 8 | 256 | 25.423 km | 648.582 km2 |
| 9 | 512 | 12.711 km | 162.146 km2 |
| 10 | 1024 | 6.356 km | 40.5364 km2 |
| 11 | 2048 | 3.178 km | 10.1341 km2 |
| 12 | 4096 | 1.589 km | 2.53352 km2 |
| 13 | 8192 | 794.456 m | 633,381 m2 |
| 14 | 16384 | 397.228 m | 158,345 m2 |
| 15 | 32768 | 198.614 m | 39,586.3 m2 |
| 16 | 65536 | 99.307 m | 9,896.58 m2 |
| 17 | 131072 | 49.654 m | 2,474.15 m2 |
| 18 | 262144 | 24.827 m | 618.536 m2 |
| 19 | 524288 | 12.413 m | 154.634 m2 |
| 20 | 1048576 | 6.207 m | 38.6585 m2 |
| 21 | 2097152 | 3.103 m | 9.66463 m2 |
| 22 | 4194304 | 1.552 m | 2.41616 m2 |
| 23 | 8388608 | 77.584 cm | 0.604039 m2 |
| 24 | 16777216 | 38.792 cm | 0.15101 m2 |
| 25 | 33554432 | 19.396 cm | 0.0377525 m2 |
| 26 | 67108864 | 9.698 cm | 0.00943811 m2 |
| 27 | 134217728 | 4.849 cm | 0.00235953 m2 |
| 28 | 268435456 | 2.424 cm | 0.000589882 m2 |
| 29 | 536870912 | 1.212 cm | 0.000147471 m2 |
<!-- table:order2res:end -->

## 4. Order-29 point encodings

<a name="resolution-discriminator"></a>

Two order-29 encodings exist and are **not distinguishable from the id
stream alone** in float64/JSON contexts:

- (a) genuinely order-29 **resolution** — real area cells at the finest
  order;
- (b) unknown-resolution locations **point-encoded at order 29** — the
  default for any raw lat/lon conversion, expected to be common.

In the packed word the suffix separates them (§1); in the decimal string it
does not (§2). Stores therefore declare which they hold.

### The `resolution` discriminator (contract)

Morton-declared zarr stores carry, in the `dggs` attrs block (§5):

```
resolution: "exact" | "point"
```

- `"exact"` — ids are true cells at their encoded order. **Grid-derived cell
  coordinates are `exact` by construction** (e.g. aggregation outputs whose
  cells come from a declared grid).
- `"point"` — ids are locations cast to order 29 with no area claim
  (location-derived id fields: raw lat/lon conversions, event streams).

Emission is **per data kind, and the writer always knows which**: a writer
producing grid cells writes `exact`; a writer converting raw coordinates
writes `point`. There is no heuristic fallback.

### The 29→24 clip rule (contract)

IEEE-754 float64 (and therefore JavaScript `Number` and plain JSON parsers)
is integer-exact only to 2^53; NESTED cell ids are float64-exact only
**through order 24**. For Number-safe consumption paths (browser-direct
readers, JSON interchange of NESTED ids):

- ids with `resolution: "point"` **are clipped on the fly to order 24** — a
  point makes no area claim, so truncating its path to order 24 loses only
  sub-40-cm location precision (§3) and never misstates coverage;
- ids with `resolution: "exact"` are **never clipped** — a genuine
  finer-than-24 area cell cannot be silently coarsened; such data takes
  other measures (server/hub-side fabrication, aggregation) before entering
  a Number-limited path.

The clip rule keys **only** on the declared `resolution` field — this is why
the discriminator exists.

## 5. Zarr DGGS convention block

<a name="dggs-attrs"></a>

**Contract.** A zarr store whose cell coordinate is packed morton words
declares, on the group holding the cell-indexed arrays:

```json
{
  "zarr_conventions": [
    {
      "schema_url": "https://github.com/espg/mortie/blob/main/docs/specification.md#dggs-attrs",
      "spec_url": "https://github.com/espg/mortie/blob/main/docs/specification.md",
      "uuid": "3e22156d-ea9e-4e01-95fe-e3809a4b41e7",
      "name": "morton-dggs",
      "description": "Packed-u64 morton (HEALPix) DGGS convention"
    }
  ],
  "dggs": {
    "name": "morton",
    "coordinate": "morton",
    "resolution": "exact",
    "...": "grid parameters (refinement level, ellipsoid, ...)"
  }
}
```

- **`name: "morton"` and `coordinate: "morton"`** — the grid name is
  distinct, *never* `name: "healpix"` + `indexing_scheme: "morton"`. A
  scheme-blind reader that recognizes `healpix` but ignores the indexing
  scheme would silently decode morton words as NESTED ids and mis-place
  every cell; an unknown grid name makes it **hard-reject with a
  diagnostic** instead. This matches moczarr's xdggs registration
  (`grid_name: "morton"`).
- **`resolution`** — the §4 discriminator, required on morton-declared
  stores.
- **Convention identity** — the `zarr_conventions` entry above is the
  **self-declared** convention record (the zarr-conventions mechanism
  supports self-declared entries). The UUID
  `3e22156d-ea9e-4e01-95fe-e3809a4b41e7` is minted once and **permanent**;
  readers may key on it. `zarr_conventions` is a list: a future upstream
  dggs-registry entry coexists alongside this one rather than replacing it.

## 6. Morton-hive store layout

**Contract.** The morton-hive layout stores one self-describing zarr per
spatial shard under a morton digit tree. The convention is versioned by the
manifest's `spec` string (`morton-hive/1`, `/2`, `/3`); **every frozen
grammar remains valid forever for stores declaring its version** — readers
discriminate by the `spec` string, never by sniffing names.

### 6.1 Common structure (all versions)

```text
{store_root}/
  morton_hive.json                 <- static manifest, versioned `spec` string
  coverage.moc                     <- optional root coverage MOC (§7.3)
  {sign+base}/{d1}/{d2}/.../       <- one decimal digit per path level
    {leaf}.zarr/                   <- vanilla zarr v3 leaf (naming per version)
    <declared sidecars>            <- stats record, sub-shardmap (per version)
```

- **Digit tree**: path components are the decimal-string components of §2 —
  the constant-width `{sign+base}` first, then one digit (`[1-4]`) per
  order. Shards live at mixed orders, so every order is a legal node. A
  manifest `path_grouping` parameter (default `1`) declares how many digits
  each component chunks; readers chunk the digit string per the manifest,
  never by assumption.
- **Node invariant**: below a product root, a node contains *only* digit
  children, `*.zarr` objects, and the declared leaf-adjacent sidecar names —
  nothing else, ever. The walker's child classification depends on the name
  set being closed.
- **Termination**: object stores have no empty prefixes and S3 LIST is
  strongly consistent, so a delimiter-LIST returning no digit children is a
  definitive "nothing finer exists". Absence is trustworthy.
- **Commit stamp**: presence is not trustworthy without one. A leaf's root
  metadata is finalized *last* (a root-group attrs update carrying the
  stamp); a `.zarr/` prefix whose root metadata lacks the stamp is debris —
  incomplete, ignorable, safe to overwrite on retry.
- **Overview flagging**: ancestor-node zarrs must carry `role: overview`
  attrs; the role is never inferred from tree position (a shallow zarr may
  be coarse *source* in a sparse region).
- **Manifest**: `morton_hive.json` at the (product) root is the reader's
  bootstrap; with it every shard path is computable arithmetically with
  zero requests. Its `spec` string versions the convention.

### 6.2 `morton-hive/1` — bare leaves

Leaf basename is the **full morton id**: `{full_id}.zarr` (e.g.
`.../1/2/3/-31123.zarr/`). Self-describing without parsing its path.

### 6.3 `morton-hive/2` — time-windowed leaves

A `/2` store's manifest carries a temporal block declaring time
encoding/units/epoch/calendar, the membership timestamp field, the window
schedule, and the append policy. A `/1` store *is* a `/2` store with
`schedule: none`.

Leaf naming:

```text
{full_id}_{window}.zarr        <- windowed leaf (schedule != none)
{full_id}.zarr                 <- bare leaf   (schedule == none only)
```

- **Separator is `_`**; it never appears in morton ids nor in window labels,
  so the split is unambiguous. **Parse rule: split on the first `_`.**
- One schedule per store; at most one leaf per (id, window); bare and
  windowed source leaves never mix in one store.

Window schedules and label grammar (lexicographic order = chronological
order within a store):

| schedule | label grammar | example label | window |
|---|---|---|---|
| `none` (default) | *(no label)* | — | unbounded; re-run replaces the leaf |
| `yearly` | `YYYY` | `2025` | `[2025-01-01T00:00Z, 2026-01-01T00:00Z)` |
| `monthly` | `YYYYMM` | `202511` | calendar month |
| `daily` | `YYYYMMDD` | `20251103` | calendar day |
| `quarterly` *(grammar-reserved, not implemented)* | `YYYYQ[1-4]` | `2025Q3` | calendar quarter |
| explicit list | opaque, `[0-9A-Za-z-]{1,32}` | `melt-2019` | declared per label in the manifest |

- **Boundaries are UTC calendar terms, half-open `[start, end)`**,
  regardless of the store's native time encoding (the temporal block
  declares the conversion). An observation stream straddling a boundary
  contributes to both windows.
- **Explicit labels are opaque**: the manifest maps each label to its
  `[start, end)`; readers never parse semantics out of a custom label. The
  charset excludes `_` by construction.
- The reserved token `all` (§6.4) is **permanently excluded from the window
  label grammar** under every schedule.
- Per-shard stats sidecars (when written): `stats_{window}.json`, or
  `stats.json` under `schedule: none`.

### 6.4 `morton-hive/3` — window-only leaf naming

Completes the axis separation: **product = root prefix (§6.5), space =
digit path, time = basename** — each identity axis in exactly one place.

```text
{window}.zarr                  <- leaf, basename = time window alone
all.zarr                       <- schedule: none (reserved token)
```

- The morton id no longer appears in the basename; it is recoverable
  arithmetically from the digit path and recorded in the leaf's stamp
  attrs / stats sidecar (`shard_key`).
- **`all` is a reserved token**: it names the `schedule: none` leaf (reads
  as "all time"), cannot collide with the digit-shaped generative labels,
  and is **excluded from the window-label grammar forever** (an explicit
  schedule must not declare a window labeled `all`).
- Sidecar naming aligns to the leaf: `{window}.stats.json` /
  `all.stats.json`.
- `/1` and `/2` stores keep their frozen grammars (§6.2, §6.3) forever;
  readers discriminate by the manifest `spec` string.

### 6.5 Product roots and the product-name grammar

A multi-product store is a **directory of stores**: each product lives under
its own human-readable root prefix `{name}/`, and a product subtree is a
*complete, unmodified* morton-hive store (bare-named manifest, MOC, digit
tree). A bare single-product store (manifest at the store root) remains
fully valid; readers distinguish the two root forms **by content** — a
manifest at the root ⇒ bare store; name-shaped prefixes ⇒ product
directory.

**Product-name grammar (contract):**

```text
product-name = 1*( lowercase-alphanum / "-" / "_" )   ; [a-z0-9_-]+
```

with the **base-component exclusion**: a product name must not match the
morton base-component grammar `-?[1-6]` (§2), so the walker's child
classification stays unambiguous. Names are URL-safe by construction (no
percent-encoding, no case-folding hazards).

### 6.6 Repr reminder for paths

Hive paths embed decimal components (§2). The §2 non-injectivity note
applies: leaf ids at order 29 could not distinguish point from area — hive
stores avoid the ambiguity structurally (shards live at coarse orders and
the `resolution` discriminator (§4) declares the cell-id kind), but no
implementation may round-trip point-ness through a path string.

## 7. Coverage MOC serializations

**Contract** (`spec: "morton-moc/1"`). Coverage is declared in three tiers;
all three share the envelope's `encoding` discriminator:
`"ranges" | "bitmap" | "full"`.

### 7.1 Stamp envelope: the tier-0 morton box

The leaf commit stamp carries the shard's **morton box** — the canonical
≤ 4-member MOC covering its occupied cells — padded to **exactly four**
decimal-string slots; the pad sentinel is **JSON `null`**. `encoding:
"full"` declares a fully occupied subtree (the shard id itself is the exact
MOC; no bitmap sidecar is written).

### 7.2 Leaf bitmap sidecar (`coverage.moc` inside the leaf)

Exact cell-order occupancy for one shard subtree:

- The raw bitmap has `4^depth` bits (`depth = cell_order - shard_order`);
  raw size is exactly `ceil(4^depth / 8)` bytes.
- **Bit convention (contract, golden-vector-pinned)**: bit `i` is the i-th
  shard-subtree cell in **ascending packed-word order** — equivalently, the
  base-4 rank of the cell's digit tail with digits `1..4` mapped to `0..3`
  — packed **MSB-first within each byte** (`np.packbits` order).
- The payload is one **zstd stream** over the raw bitmap. The zstd *stream
  format* is contract; the compression **level is non-normative** (any
  level decodes identically).
- A decoder must reject a payload whose decompressed size is not exactly
  the raw bitmap size — never zero-pad or truncate (a partial cell set is a
  false negative).

### 7.3 Root coverage MOC (`coverage.moc` at the store/product root)

A JSON envelope with `encoding: "ranges"` listing shard-order coverage:

- A range is an **inclusive `[first, last]` run of same-order cells within
  one base cell, consecutive in base-4 digit-tail rank** (the same
  ascending packed-word order as §7.2) — **range ordering is contract**
  (golden-vector-pinned).
- Endpoints are **decimal strings** (§2), never JSON numbers: packed words
  exceed 2^53 and would be silently mangled by float-based JSON parsers.
- Carrier fields (`source`, `generated_at`, optional `time_range`) are
  informative cache metadata; the ranges are a regenerable cache of the
  leaf-stamp truth.

## 8. Frozen for 1.x

The 1.x contract guarantees, immutable within the major version:

- the §1 bit layout, order range 0–29, canonical zero-fill, unsigned
  storage, and the raw-sort Z-order property;
- the §2 decimal grammar, its render-only status, and the emit conventions
  (strings display / `uint64` storage / capped legacy `i64` escape hatch);
- the §4 `resolution` discriminator values and the 29→24 clip rule's
  point-only scope;
- the §5 convention identity (UUID) and the `name: "morton"` /
  `coordinate: "morton"` declaration;
- the §6 hive grammars — `/1`, `/2`, `/3` each frozen for stores declaring
  its `spec` string — including the `_` split rule, the window-label
  grammars, the `all` reserved token, the product-name grammar, and the
  node invariant;
- the §7 coverage contracts: the 4-slot null-padded box, the `encoding`
  discriminator values, the bitmap bit convention, and the root-MOC range
  ordering (zstd level and other codec parameters stay non-normative).

Extensions (new schedules, new `spec` versions, new encodings) are additive
under new discriminator values; existing stores never reparse under new
rules.
