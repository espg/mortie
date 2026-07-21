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
4. [Order-29 points: encoding-carried kind and the decimal-parse tie-break](#4-order-29-points)
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
carried by the suffix range, not by any external flag — the load-bearing
convention §4 states normatively — and the only place the two kinds can
collide is the order-29 decimal string, resolved by the §4 parse tie-break.

## 2. Decimal string representation

**Contract.** The decimal string is the **render/interchange** external form of a
morton index (paths, logs, inventories, display). Packed `uint64` words are
the storage and compute form; output types are never data-dependent:

- **strings** for display/interchange at every order;
- **packed `uint64`** for storage and compute;
- legacy signed `i64` only via the explicit, capped `to_legacy_i64()` escape
  hatch.

Grammar:

```text
morton-decimal = ["-"] base-digit *order-digit [kind-suffix]
base-digit     = "1" / "2" / "3" / "4" / "5" / "6"
order-digit    = "1" / "2" / "3" / "4"
kind-suffix    = "p"    ; POINT ids only; render/interchange form only
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
- **Kind suffix `p`** (espg-ruled in review, 2026-07-21): the render form
  MAY carry a terminal `p` on **point** ids (e.g. `-62…21p`), and only
  there — a `p` is legal solely on a full order-29 string, because points
  exist only at order 29. **Paths never carry it**: points don't live in
  paths, and the §6 path grammar is unchanged. Rationale: no letter occurs
  anywhere in the base grammar (sign, base digit, digits `1-4`), so a
  letter suffix is maximally distinguishable, greppable, and inert in
  shells, markup, and URLs; `*` and a terminal `.` were considered and
  rejected (glob/markdown hazards; trailing-dot stripping).

### Order-29 kind marking and the unmarked tie-break

With the kind suffix, the decimal round-trip is **lossless for both
kinds** — the normative contract: an area word renders unmarked and parses
back to itself; a point word renders `p`-marked and parses back to itself.
The `p` emit/accept is implemented in
[PR #121](https://github.com/espg/mortie/pull/121) (issue #120), with
golden vectors. The residual ambiguity
is only the **unmarked** order-29 string — path components and legacy
renders — which denotes both kinds and parses as the **area** word (the
normative §4 tie-break; every pre-suffix string is an area context, so the
rule is fully backward compatible).

## 3. Resolution table

Informative (the formulas are contract; the rendered values are derived).
`nside = 2^order`; **both columns derive from one Earth model — the exact
HEALPix sphere at mean radius `R = 6371.0088 km`.** Every order-*k* HEALPix
cell has identical area `4πR² / (12 · 4^order)` (HEALPix is equal-area by
construction); the **cell scale** is the square root of that area — the RMS
cell spacing `sqrt(4πR² / (12 · 4^order))`. The table below is regenerated
from these formulas and pinned by `mortie/tests/test_spec_page.py` so it
cannot drift.

**Note — code and page unified.** These are the **normative, sphere-derived**
values, and `mortie.tools.order2res` now derives from the same sphere:
`order2res(order) = sqrt(4πR² / (12 · 4^order))` with the single
`mortie.tools.EARTH_RADIUS_KM = 6371.0088` constant. Its consumers
(`res2display` and the buffer-pad computation in
`tests/test_coverage_boundary.py`) therefore read the cell-scale column
below directly. This replaced the historical flat constant
`111 km/deg × 58.6323 × 0.5^order` (an implied sphere `R ≈ 6366 km`), a
behavioral change of ~0.2% at every order, per
[mortie #119](https://github.com/espg/mortie/issues/119).

<!-- table:order2res:begin -->
| order | nside | cell scale | cell area |
|---|---|---|---|
| 0 | 1 | 6,519.623 km | 4.25055e+07 km2 |
| 1 | 2 | 3,259.812 km | 1.06264e+07 km2 |
| 2 | 4 | 1,629.906 km | 2.65659e+06 km2 |
| 3 | 8 | 814.953 km | 664,148 km2 |
| 4 | 16 | 407.476 km | 166,037 km2 |
| 5 | 32 | 203.738 km | 41,509.3 km2 |
| 6 | 64 | 101.869 km | 10,377.3 km2 |
| 7 | 128 | 50.935 km | 2,594.33 km2 |
| 8 | 256 | 25.467 km | 648.582 km2 |
| 9 | 512 | 12.734 km | 162.146 km2 |
| 10 | 1024 | 6.367 km | 40.5364 km2 |
| 11 | 2048 | 3.183 km | 10.1341 km2 |
| 12 | 4096 | 1.592 km | 2.53352 km2 |
| 13 | 8192 | 795.852 m | 633,381 m2 |
| 14 | 16384 | 397.926 m | 158,345 m2 |
| 15 | 32768 | 198.963 m | 39,586.3 m2 |
| 16 | 65536 | 99.482 m | 9,896.58 m2 |
| 17 | 131072 | 49.741 m | 2,474.15 m2 |
| 18 | 262144 | 24.870 m | 618.536 m2 |
| 19 | 524288 | 12.435 m | 154.634 m2 |
| 20 | 1048576 | 6.218 m | 38.6585 m2 |
| 21 | 2097152 | 3.109 m | 9.66463 m2 |
| 22 | 4194304 | 1.554 m | 2.41616 m2 |
| 23 | 8388608 | 77.720 cm | 0.604039 m2 |
| 24 | 16777216 | 38.860 cm | 0.15101 m2 |
| 25 | 33554432 | 19.430 cm | 0.0377525 m2 |
| 26 | 67108864 | 9.715 cm | 0.00943811 m2 |
| 27 | 134217728 | 4.857 cm | 0.00235953 m2 |
| 28 | 268435456 | 2.429 cm | 0.000589882 m2 |
| 29 | 536870912 | 1.214 cm | 0.000147471 m2 |
<!-- table:order2res:end -->

## 4. Order-29 points

<a name="point-kind"></a>

**Contract. Kind is carried by the encoding itself — never by store or
array metadata.** The packed word's suffix region (§1) *is* the kind:

- suffix `0..=47` decodes as an **area** word — an exact cell at its
  encoded order, at **every** order 0–29. An order-29 area word is a
  genuine order-29 cell; nothing is unrepresentable.
- suffix `48..=63` decodes as an order-29 **point** — a location with no
  area claim, full stop.

There is no `resolution`/kind field in the §5 attrs block and readers never
consult a declaration: two encodings, two meanings, one word. Mixed content
— exact cells at any orders alongside order-29 points — is well-formed in a
single coordinate by construction, because each word carries its own kind.
*(Provenance: espg-ratified 2026-07-21 on the PR #118 review, superseding
the drafted declaration-based designs.)*

### The decimal-parse tie-break (contract)

Packed words never collide across kinds — the suffix ranges are disjoint.
The **one ambiguity in the convention** is the decimal repr at order 29: a
full order-29 string (base component + 29 digits) denotes *both* the
order-29 area cell and the max-encoded point on the same path, because the
string carries path only, never kind (§2). Strings at orders 0–28 denote
area cells alone (points exist only at order 29). The normative tie-break:

> **A `p`-marked string yields the POINT word; an unmarked string always
> yields the AREA word.** For an order-29 path with final two (stored)
> tuple values `t28`, `t29`: unmarked ⇒ the area word (suffix
> `28 + t28·5 + t29 + 1`); `p`-marked ⇒ the point word (suffix
> `48 + t28·4 + t29`). Round-trip identity holds for both kinds; the
> unmarked-string rule is the tie-break for the one truly ambiguous form
> (§2), and since every pre-suffix string is unmarked it is fully
> backward compatible.

In channels that strip or cannot carry the suffix (paths above all),
point-ness does not survive the string: those channels carry the packed
word when kind matters. The unmarked tie-break is what mortie's parser has
always implemented, golden-pinned by `mortie/tests/test_spec_page.py`; the
`p` emission/acceptance is implemented in
[PR #121](https://github.com/espg/mortie/pull/121) (issue #120), with
golden vectors.

### Points at coarser levels (informative)

Membership of a point in a coarser cell is the ordinary truncation
(coarsen / `clip2order`: drop the path tail below the target order) — a
**transient cast** computed where needed, never a stored re-encoding.

### Viewer-side float64 casts (informative)

IEEE-754 float64 (JavaScript `Number`, plain JSON parsers) is
integer-exact only to 2^53, which covers NESTED ids only through order 24.
A display layer in such a runtime (e.g. the gridlook viewer) may
**transiently** cast point ids to order ≤ 24 for Number safety — an
implementation detail of that display layer, not encoding semantics; other
viewers and future runtimes need no such cast. Area cells are never
coarsened this way: coarsening an area cell changes the labelled thing
(that is aggregation, cf. zagg D24) — servers fabricate or aggregate
instead.

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
- There is **no kind/`resolution` field**: point-vs-area kind is carried by
  the word encoding itself (§4), never by attrs.
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
- The reserved token `all` (§6.4) is **excluded from the window-label
  grammar going forward**: no store written under any spec version may
  declare an explicit window labeled `all`. In `/2` the token has no
  structural role — the `schedule: none` leaf is the bare `{full_id}.zarr`,
  not `all.zarr` (that is a `/3` construct) — so the reservation is a
  forward-going constraint on new writers, not a retroactive narrowing of the
  frozen `/2` opaque grammar. A pre-existing `/2` store whose manifest
  happened to declare an `all` label stays readable under its frozen grammar:
  `/2` labels are opaque and manifest-resolved, so `all` there is just
  another custom label.
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
- **`all` is a reserved token**: it names the `/3` `schedule: none` leaf
  (reads as "all time"), cannot collide with the digit-shaped generative
  labels, and is **excluded from the window-label grammar going forward** (no
  store, any spec version, may declare an explicit window labeled `all`). The
  token is structural only in `/3`; §6.3 records how the same forward-going
  reservation applies to `/2`, which has no `all` leaf.
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
product-name = 1*192( lowercase-alphanum / "-" / "_" )   ; [a-z0-9_-]{1,192}
```

with the **base-component exclusion**: a product name must not match the
morton base-component grammar `-?[1-6]` (§2), so the walker's child
classification stays unambiguous. Names are URL-safe by construction (no
percent-encoding, no case-folding hazards).

- **Length 1–192 characters.** The charset is single-byte ASCII, so 192
  chars = 192 bytes. Derivation: the POSIX filename-component ceiling is 255
  bytes; a D23 downloader that materializes the tree locally shares the
  product's path component with a 13-character immutable-provenance
  decoration (`{name}+{catalog-hash}/` — a `+` plus a 12-hex catalog
  fingerprint), leaving a hard ceiling of `255 − 13 = 242`. The 192 cap sits
  50 characters under that ceiling and keeps total-path budgets comfortable
  (~400 chars of the 1,024-byte S3 total-key budget, and macOS `PATH_MAX`, at
  order-24 digit depth under realistic prefixes).

### 6.6 Repr reminder for paths

Hive paths embed decimal components (§2). The §2 non-injectivity note
applies: leaf ids at order 29 could not distinguish point from area — hive
stores avoid the ambiguity structurally (shards live at coarse orders, and
in-store kind rides the packed words themselves, §4) — and any parsed path
string resolves by the §4 tie-break; no implementation may round-trip
point-ness through a path string.

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

A JSON envelope with `encoding: "ranges"` listing shard-order coverage.

**Required keys** (a reader validates the set, never infers it):

- **`spec`** — `"morton-moc/1"` (§7 header), tying the envelope to this
  serialization;
- **`encoding`** — `"ranges"` for this tier;
- **`order`** — integer, the shard order of every cell in `ranges` (the
  common order the decimal endpoints share);
- **`ranges`** — the list of `[first, last]` runs below.

`source`, `generated_at`, and optional `time_range` are informative carrier
fields. Field semantics:

- A range is an **inclusive `[first, last]` run of same-order cells within
  one base cell, consecutive in base-4 digit-tail rank** (the same
  ascending packed-word order as §7.2) — **range ordering is contract**
  (golden-vector-pinned).
- Endpoints are **decimal strings** (§2), never JSON numbers: packed words
  exceed 2^53 and would be silently mangled by float-based JSON parsers.
  They are always **unmarked area words** at the shard order, never
  `p`-marked: points live only at order 29 and cannot be range endpoints.
- The carrier fields above are informative cache metadata; the ranges are a
  regenerable cache of the leaf-stamp truth.

## 8. Frozen for 1.x

The 1.x contract guarantees, immutable within the major version:

- the §1 bit layout, order range 0–29, canonical zero-fill, unsigned
  storage, and the raw-sort Z-order property;
- the §2 decimal grammar, its render/interchange status, and the emit conventions
  (strings display / `uint64` storage / capped legacy `i64` escape hatch);
- the §4 encoding-carried kind convention (suffix `0..=47` = area, exact
  at every order; `48..=63` = order-29 point) and the decimal parse rules
  (`p`-marked string ⇒ point word; unmarked string ⇒ area word — the
  tie-break; the `p` kind suffix is render/interchange-only and never
  appears in paths);
- the §5 convention identity (UUID) and the `name: "morton"` /
  `coordinate: "morton"` declaration;
- the §6 hive grammars — `/1`, `/2`, `/3` each frozen for stores declaring
  its `spec` string — including the `_` split rule, the window-label
  grammars, the `all` reserved token (structural in `/3`; a forward-going
  window-label exclusion across all spec versions, §6.3/§6.4), the
  product-name grammar (charset, base-component exclusion, and the 1–192
  character length cap), and the node invariant;
- the §7 coverage contracts: the 4-slot null-padded box, the `encoding`
  discriminator values, the bitmap bit convention, and the root-MOC range
  ordering (zstd level and other codec parameters stay non-normative).

Extensions (new schedules, new `spec` versions, new encodings) are additive
under new discriminator values; existing stores never reparse under new
rules.
