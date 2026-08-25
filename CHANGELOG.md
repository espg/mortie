# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- **BREAKING: `MortonIndexScalar` is renamed `MortonWord`, exported flat,
  constructs from the decimal label, and grows strict `.decimal` /
  `.order` / `.base_cell` accessors** (issue #152; pre-1.0, no alias). The
  scalar word type is now `mortie.MortonWord` (`decimal_to_word`'s
  `dtype=` escape spells it the same way). A `str` argument parses as a
  decimal Morton label through `decimal_to_word` (point-suffix grammar
  included): `MortonWord("4331422412232")` is the cell that displays as
  `4331422412232`, where the inherited `numpy.uint64` constructor used to
  read the label as a base-10 *packed word* and silently construct the
  wrong cell. An invalid label raises `ValueError` at the boundary,
  naming the input and the grammar, and bytes-like input is refused with
  a pointed `TypeError` (decode to `str` for a label, pass an `int` for a
  word) as ambiguous rather than guessed at — numpy reads `bytes` as a
  base-10 word and `bytearray` as a raw buffer, and neither reading is
  the label. The accessors are **strict data queries**: a word that
  decodes to no legal cell (the empty sentinel included) raises a pointed
  `ValueError` naming the word, so invalid data does not propagate;
  `.base_cell` is so named because `numpy.generic` already owns `.base`.
  `int` / `numpy.uint64` construction, arithmetic, and the
  lazy/never-raise display (`"<NA>"` / `"<invalid 0x...>"`) are
  unchanged. Because the type pickles by name (`__reduce__` rebuilds the
  wrapper), **pickles of `MortonIndexScalar` written by prior releases no
  longer unpickle** — the class that name refers to is gone. That is a
  deliberate pre-1.0 break: no alias or shim is provided, so re-emit any
  persisted scalars (or store the packed `int` / `uint64`, which is
  version-independent). Adoption at mortie's own morton-scalar return
  sites is deferred to 1.1 (issue #215).

- **BREAKING: one polymorphic function per operation — the plural batch names
  are removed** (issue #187, ruled 2026-08-19). Every scalar/batch pair now has
  **one** public entry point: the input shape (or the keyword-only `offsets=`)
  selects the form, and the redundant sibling is gone outright — no deprecation
  shims, no aliases. This lands ahead of the 1.0 release; migration is one line
  per name:

  | removed | call instead |
  |---|---|
  | `mocs_to_orders(values, offsets, order, max_cells)` | `moc_to_order(values, order, max_cells, offsets=offsets)` |
  | `mocs_and(a, values, offsets)` | `moc_and(a, values, offsets=offsets)` |
  | `mocs_intersect(a, values, offsets)` | `moc_intersects(a, values, offsets=offsets)` |
  | `common_ancestors(values, offsets)` | `common_ancestor(values, offsets=offsets)` |
  | `tocs_reduce(words, offsets)` | `toc_reduce(words, offsets=offsets)` |
  | `decimals_to_words(arr)` | `decimal_to_word(arr)` (array in, array out) |
  | `children_of(words, order, max_cells)` | `generate_morton_children(words, order, max_cells=max_cells)` |
  | `from_wkbs(blobs, ...)` | `from_wkb(blobs, ...)` — see below; the batch is a `list` / `tuple` / object-`ndarray`, so materialize anything else first (`list(gen)`, `series.to_numpy()`, `arr.astype(object)`) |
  | `morton_coverage_moc(lats, lons, ...)` | `values, _ = polygons_to_morton_mocs(lats, lons, [0, len(lats)], ...)` for one ring — the batch-native call returns the ragged `(values, out_offsets)` pair, so the one ring's MOC is `values`, not the return itself; for multipart/holes see the note below |
  | `mortie.arrow.from_wkbs(column, ...)` | `mortie.arrow.from_wkb(column, ...)` (renamed with the core) |

  Notes on the two non-mechanical rows. **`from_wkb`** is now polymorphic with
  no `batch=` flag: `offsets=` given means a packed binary column (a `uint8`
  values buffer plus arrow list offsets, sliced zero-copy); `list` / `tuple` /
  object-`ndarray` means a sequence of blobs, each coerced as the scalar form
  accepts; `bytes` / hex `str` / `bytearray` / `memoryview` / `uint8`-ndarray
  without `offsets` means one blob. The dispatch is exhaustive by design — it
  does not sniff any wider — so a container `from_wkbs` used to iterate but
  this rule does not name is a `TypeError`: a pandas `Series`, a generator and
  a bytes-dtype (`S`) array all need materializing to one of the three batch
  spellings first (`series.to_numpy()`, `list(gen)`, `arr.astype(object)`), or
  packing into the `offsets=` column form. Its `moc` argument is a tri-state:
  the default (`None`) keeps both historical behaviours — flat cover for one
  blob, ragged MOC pair for a batch — `moc=True` works everywhere, and an
  explicit `moc=False` on a batch raises (there is no ragged flat-cover
  kernel). Migrating a positional `from_wkbs(blobs, order, tol)` call needs
  `tolerance=` spelled as a keyword, since `from_wkb`'s third positional is
  `moc` — and `moc` is type-guarded to `bool` / `None`, so that migration
  raises `TypeError` naming the parameter, the received value and type, and
  the hazard, instead of binding the tolerance to `moc` and silently
  dropping it. **The MOC coverer** is batch-native: `polygons_to_morton_mocs`'
  ragged signature has no scalar shape to collapse into, so the plural
  survives there and the scalar `morton_coverage_moc` is the name that
  retired (issue #187 P0, ruled). Its entries are single rings, so a
  multipart/hole ring-set is covered instead through `from_wkb(blob,
  moc=True, order=...)` (backend-free, WKB bytes in), `from_geometry` /
  `from_wkt` with `moc=True` (an `order`, but a geometry backend is
  required), or `mortie.Moc` / `Moc.from_polygon` (backend-free, ring arrays
  or GeoJSON in, but no `order` — it covers at the default finest order).

  Refusals that used to name a retired delegate now name the surviving
  function (`toc_reduce of an empty segment`, `generate_morton_children only
  refines`). The batch kernels live on as private functions in
  `mortie.batch` / `mortie._toc` / `mortie.morton_index` / `mortie.coverage`;
  private names carry no compatibility promise.

- **BREAKING (small): `norm2mort` and `mort2norm` keep a length-1 array an
  array** (issue #187). They used to squeeze a one-element input to a bare
  scalar, which is the opposite of the array-in/array-out rule the polymorphic
  API is built on, and silent — the caller who passed an array got back
  something that could not be indexed. The form now follows the **input rank**
  in both: `norm2mort` returns a scalar only when both `normed` and `parent`
  are scalars, and `mort2norm` only for a scalar or 0-d word (its `order`
  return is a plain `int` either way). The pair is documented as exact
  inverses, so they move together — fixing only the forward direction would
  have left `mort2norm(norm2mort([n], [p], o))` handing back scalars for an
  array round trip. `np.atleast_1d(norm2mort(...))` at a call site becomes a
  no-op rather than a fix. What a length-1 result stops doing, in order of how
  hard it bites: it is **no longer hashable** (`{norm2mort([n], [p], o): …}`
  raises `TypeError`, so a morton word used as a dict key breaks outright),
  it **formats as `[123]` rather than `123`** (an f-string in a path or a log
  line changes silently), and `int(...)` on it raises a `DeprecationWarning`
  and still returns on the numpy this release tests against (2.2.2) — it
  errors only under `-W error::DeprecationWarning`, and on a future numpy.
  Take `[0]` in all three cases.

- **BREAKING: every word-valued scalar return is `np.uint64`** (issue #187).
  A mortie *word* — morton or toc — now means one Python type on every entry
  point. `time2toc`, `span2toc`, `toc_merge` and `toc_reduce` returned a plain
  Python `int` for scalar input and now return `np.uint64`, matching
  `norm2mort`, `common_ancestor` / `moc_min` and `decimal_to_word`, which
  already did. Values are bit-identical; only the type changes. What this
  breaks: `type(w) is int` / `isinstance(w, int)` checks (`np.uint64` does not
  subclass `int`), `json.dumps` of a bare word (use `int(w)`), `int`'s own
  methods — `w.to_bytes(8, "little")`, `w.bit_length()` and
  `w.as_integer_ratio()` now raise `AttributeError`, and unlike `json.dumps`
  they fail with no hint attached, so reach for `int(w).to_bytes(...)`
  (`w.bit_count()` survives; numpy has its own) — and arithmetic
  expectations: `uint64` **wraps at 2\*\*64** instead of promoting to a big
  int, and mixing it with a Python `float` gives `float64`. An *arithmetic*
  wrap is observable — numpy emits `RuntimeWarning: overflow encountered in
  scalar add` *before* wrapping, so under `-W error::RuntimeWarning` or
  `np.seterr(over="raise")` it raises instead — but a bit shift off the top
  of the word (`w << 1`) truncates silently, with no warning to catch.
  Comparisons, hashing, dict keys, f-strings and `int(w)` are unaffected.

  Deliberately **not** unified, because they are not words: times in ns
  (`toc2time`, `from_datetime64`, `from_gps_ns`, `to_gps_ns`), HEALPix orders
  (`infer_order_from_morton`), UNIQ cell ids (`geo2uniq`, `norm2uniq`,
  `unique2parent` — a different encoding, and inconsistent among themselves
  today), and the explicit return-shape escapes `decimal_to_word(dtype=int)` /
  `dtype=MortonWord` and the private `_decimal_to_word`. The toc
  set-algebra kernels (`toc_normalize`, `toc_and`) always return arrays, so
  there is no scalar to unify.

  To be clear about the one case that *looks* like an exclusion and is not:
  element access on the object layer — `MortonIndexArray[0]`, iteration,
  `.take` / `.unique` / `.tolist()[0]`, the arrow skin, and a pandas `Series`
  of the extension dtype — hands back a `MortonWord`. That **is** a
  `uint64` word: it subclasses `np.uint64` and overrides nothing but its
  string presentation (`str` / `repr` / `format` give the decimal-morton
  spelling, and `__reduce__` carries that identity through pickle), so its
  value, arithmetic, comparisons and hashing are `np.uint64`'s. Those paths
  therefore **satisfy** this unification rather than opting out of it. The
  predicate to write at a call site is `isinstance(w, np.uint64)`, which
  holds for every word from every entry point; `type(w) is np.uint64` is the
  stricter pin the *bare* functions above must meet, and the test suite holds
  them to it.

- **BREAKING: the numpy floor is now `numpy>=2`** (issue #187, ruled
  2026-08-19). NEP 50 is what makes the `np.uint64` word semantics above
  correct: below numpy 2 a word's arithmetic promotes to `float64`, which is
  inexact above 2^53 while mortie words run near 2^62, and bitwise ufuncs
  against a Python `int` raise `TypeError` outright. The previous `>=1.20` was
  an untested declaration — every CI job installs numpy unpinned, so the whole
  matrix has only ever run numpy 2 — and 1.0 is the honest moment to state the
  floor the package actually supports. Both conda envs in the tree track it —
  the dev `environment.yml` and `binder/environment.yml`. The behaviour the
  floor exists for is now pinned by tests, and so is the declaration itself,
  so neither a downgrade nor a quiet edit of the floor passes silently.

- **`validate_morton` checks every element's order** (issue #187). It is marked
  batch vectorized, and the optional `order` argument is now compared against
  **every** word rather than against `depths[0]` alone — a mixed-order array
  used to pass validation on the strength of its first element while the rest
  went unchecked (the decode itself always ran per element). The refusal names
  the lowest-index offender and its own order; the message for a **scalar**
  word is unchanged (no `(word i of n)` suffix). That suffix follows the
  **input rank**, matching `norm2mort` / `mort2norm` above: a length-1 array
  is an array, so it now reads `(word 0 of 1)` where it used to be squeezed
  into the scalar message. An **empty input now returns `True`** for any `order` rather than
  raising `IndexError` — no word disagrees, so the reduction is vacuously true
  (`np.all([])`), and the batch form stays a no-op on a legal empty column
  instead of refusing it. The old `IndexError` came from indexing `depths[0]`
  and was an accident of the first-element check, not a verdict. A non-scalar
  `order` is now refused with a `TypeError` naming it: the per-element
  comparison would otherwise broadcast an array-valued `order` into an
  undocumented per-element expectation (and print the whole array as
  "expected"), where the scalar comparison it replaced raised. One order,
  checked against every word — use `orders_of` for per-element orders.

## [0.9.11] - 2026-08-24

- workspace split: extract mortie-core (issue #200) ([#207](https://github.com/espg/mortie/pull/207)) by @espg
- Spec: normative toc word grammar, frozen for 1.x (issue #193) ([#206](https://github.com/espg/mortie/pull/206)) by @espg
- Toc object: temporal coverage composing like Moc (issue #198) ([#199](https://github.com/espg/mortie/pull/199)) by @espg


- **Spec: normative section for the toc word grammar, frozen for 1.x**
  (issue #193). Docs-only, no kernel changes: `docs/specification.md` gains
  §11 "The packed 64-bit toc word", appended after the freeze list —
  bit layout, the 1850 GPS-aligned epoch and its UTC boundary convention
  (leap steps cited to IERS Bulletin C), the outward-rounding encode law,
  decode semantics and the exact valid-domain characterization, unsigned
  sort order, the semilattice merge law with its valid-domain scope, the
  window-predicate conservatism directions, and golden conformance vectors
  pinned against the live kernels by `mortie/tests/test_spec_toc.py`.
  "Frozen for 1.x" stays §10 — number and anchor untouched, new sections
  append after it — and gains the toc bullet. External stores citing the
  grammar (zagg's `zagg-toc/1`, token `mortie-toc/1`) now resolve to a
  versioned-normative section instead of the API page.

## [0.9.10] - 2026-08-19

- Moc object: geometry-first coverage API (issue #196) ([#197](https://github.com/espg/mortie/pull/197)) by @espg
- Close the vertex-point-touch gap in the closed-set contract (follow-up to #107) ([#148](https://github.com/espg/mortie/pull/148)) by @espg


- **BREAKING: `mortie.toc` is the `Toc` constructor, not a submodule** (issue
  #198). `mortie/toc.py` is now `mortie/_toc.py`, which frees the `mortie.toc`
  name for a callable — the same move issue #196 made for `mortie.moc`.
  **Statement-form `import mortie.toc` and `from mortie.toc import x` break at
  this rename** — the module does not exist any more, and no import-system
  shim is possible because a callable cannot also be a module. The flat
  package names are unchanged and are the supported spelling:
  `mortie.time2toc`, `mortie.span2toc`, `mortie.toc2time`, `mortie.toc_merge`,
  `mortie.toc_reduce`, `mortie.tocs_reduce`, `mortie.toc_is_range`,
  `mortie.toc_overlaps`, `mortie.toc_contains`, `mortie.from_datetime64`,
  `mortie.to_datetime64`, `mortie.from_gps_ns`, `mortie.to_gps_ns` — the four
  grid/epoch constants, which previously lived only on the submodule, are now
  flat too: `mortie.Q_START_NS`, `mortie.Q_END_NS`, `mortie.TOC_MAX_NS`,
  `mortie.GPS_EPOCH_NS` — and `mortie.toc_normalize` / `mortie.toc_and` are
  **new in this release, flat from the start**: they never had a submodule
  spelling, so they are not in the shim's roster and
  `mortie.toc.toc_and` was never reachable. Attribute access to the old names
  (`mortie.toc.toc_merge`, `mortie.toc.Q_START_NS`) still resolves through a
  migration shim for **one minor version**, emitting a `DeprecationWarning`
  on each access (deduplication is left to the standard warnings filters);
  the attributes then drop.

- **`Toc`: a time-first temporal coverage object** (issue #198).
  `toc("2020-01-01", "2021-06-01")` builds a temporal coverage from ISO
  strings or `datetime64` instants and pairs (via `time2toc` / `span2toc`,
  broadcasting), a `uint64` toc word array, or anything exposing the new
  `__toc_words__()` interchange dunder. The canonical form is a word **set**
  — `toc_normalize`'s sorted maximal merges, kept eagerly and stored
  read-only, so `==` and `hash()` are well defined and gappy coverage keeps
  its gaps (the constructor docstring states the one-way lossy-toward-
  coverage act: subsumed instants are absorbed and not recoverable from the
  cover). Methods are `overlaps`, `contains`, and `intersection` / `&` —
  **every public method a single delegation to `toc_and`**, the one set
  operation the issue #177 call-site audit ruled in, pinned mechanically by
  the same delegation test machinery as `Moc` (now shared in
  `mortie/tests/delegation.py`); union is construction
  (`Toc(np.append(a.words, b.words))`) and difference/xor deliberately do
  not ship. The predicates are documented as envelope algebra with a
  conservative-direction table; `repr` prints the span/instant counts, the
  outward-rounded UTC extent, and the covered duration. See
  [docs/api/toc_object.md](docs/api/toc_object.md).

- **Toc set algebra: `toc_normalize` and `toc_and`** (issues #177 / #198).
  The two entries the issue #177 call-site audit ruled in, both new public
  flat names. `toc_normalize(words)` is the **canonical cover form**: the
  sorted word set with the same decoded coverage as the input — ranges
  coalesce iff their decoded half-open envelopes overlap or abut exactly (a
  surviving gap is never bridged, however small, because outward rounding
  only shrinks apparent gaps), a timestamp a range subsumes is absorbed, and
  a free instant survives bit-identical. `toc_and(a, b)` is the **one set
  operation** over that form: both operands canonicalized, then a sorted
  sweep emitting `[max(starts), min(ends))`, with a timestamp surviving iff
  genuinely covered on both sides. Conservative directions: normalize is
  coverage-identical with no rounding arm anywhere (merged bounds are
  min/max of on-grid values); intersection is **exact by grid closure** —
  the max of two starts stays on the 2^31 ns start grid and the min of two
  ends on the 2^32 ns end grid — and never under-covers the true
  intersection, over-covering only by the operands' own inherited quantum.
  Union needs no operator (concatenate, then `toc_normalize`); **difference
  and xor deliberately do not ship**, because conservative covers
  *under*-cover on subtraction and no audited call site exists. Both release
  the GIL and carry `toc_merge`'s scope — an arbitrary bit pattern is
  garbage in, garbage out, deterministically.

- **BREAKING: `mortie.moc` is the `Moc` constructor, not a submodule** (issue
  #196). `mortie/moc.py` is now `mortie/_moc.py`, which frees the `mortie.moc`
  name for a callable. **Statement-form `import mortie.moc` and
  `from mortie.moc import x` break at this rename** — the module does not exist
  any more, and no import-system shim is possible because a callable cannot
  also be a module. The flat package names are unchanged and are the supported
  spelling: `mortie.compress_moc`, `mortie.moc_to_order`, `mortie.moc_or`,
  `mortie.moc_and`, `mortie.moc_intersects`, `mortie.moc_minus`,
  `mortie.moc_xor`, `mortie.moc_not`, `mortie.moc_min`,
  `mortie.common_ancestor`, `mortie.split_base_cells`. Attribute access to the
  old names (`mortie.moc.moc_and`) still resolves through a migration shim for
  **one minor version**, emitting a `DeprecationWarning` on each access
  (deduplication is left to the standard warnings filters); the attributes
  then drop. Enumeration of the consumers showed top-level imports are the
  norm, so the blast radius is small — but this is a break, not a deprecation,
  and this is the notice.

- **`Moc`: a geometry-first coverage object** (issue #196). `moc(geojson)`
  builds a multi-order coverage — no `order` argument, coarse interior and fine
  boundary by default — from a GeoJSON `Feature` / `FeatureCollection` /
  `Polygon` / `MultiPolygon` mapping (parsed without shapely), bare
  `[lon, lat]` ring arrays, an existing `uint64` word array, or anything
  exposing the new `__morton_moc__()` interchange dunder. Words are normalized
  eagerly with `compress_moc` and stored read-only, so `==` and `hash()` are
  well defined and construction is deterministic (same input + same version →
  the same words, byte for byte). The API mirrors MOCpy vocabulary where it
  applies — `from_polygon`, `union` / `|`, `intersection` / `&`, `difference` /
  `-`, `symmetric_difference` / `^`, `contains`, `within`, `intersects` — plus
  `.to_order(order)` for the fixed-order cast, `len()` / iteration over the words,
  and a `repr` that prints the cell count and the orders actually present.
  **Two layers, and they stay separate**: the free `moc_*` kernel functions are
  the array/batch layer (unchanged, un-deprecated, zero wrapping cost) and the
  object is ergonomics — every `Moc` method is a *single* delegation to a
  kernel function, pinned mechanically by a test over the source. The
  predicates are documented as cover algebra rather than polygon algebra, with
  a conservative-direction table saying which way each answer can err near a
  boundary. See [docs/api/moc_object.md](docs/api/moc_object.md) for the MOCpy
  crosswalk.

## [0.9.9] - 2026-08-16

- Segmented toc reduce: tocs_reduce (issue #177 v1) ([#192](https://github.com/espg/mortie/pull/192)) by @espg
- small fixes: fold the standing #185 review threads ([#191](https://github.com/espg/mortie/pull/191)) by @espg
- Example notebook for the toc module (issue #180) ([#184](https://github.com/espg/mortie/pull/184)) by @espg
- small fixes: moc.rs densify shift wrap (#161) and the batch memory posture's missing input copy (#162) ([#185](https://github.com/espg/mortie/pull/185)) by @espg


- **Segmented toc reduce: `tocs_reduce`** (issue #177)
  ([#192](https://github.com/espg/mortie/pull/192)). The ragged sibling of
  `toc_reduce`: ragged `(words, offsets)` in arrow list layout in, one merged
  `uint64` out per group, GIL released and rayon across groups. Result `i` is
  bit-identical to `toc_reduce` on group `i` alone — same semilattice join,
  same instant preservation, same fold-tree independence — over
  encoder-produced words, the scope `toc_merge` already carries (an arbitrary
  bit pattern is garbage in, garbage out, and the two fold trees may then
  differ, each deterministically). An **empty group is
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
