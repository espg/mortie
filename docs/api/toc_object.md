# mortie.Toc — the temporal coverage object

`mortie.toc(...)` builds a `Toc`: a temporal coverage as an object, so that
gappy time coverage reads as time.

```python
from mortie import toc

when = toc("2020-01-01", "2021-06-01")
assert store_toc.overlaps(when)
sliver = store_toc & when            # the canonical cover of the overlap
```

## The two-layer rule

mortie's temporal-coverage surface is two layers and stays that way, the same
split [the spatial object](moc_object.md) documents:

- **The kernel functions are the array/batch layer.** The free `toc_*`
  functions on [the toc kernel page](toc.md) are words in, words out,
  unchanged and un-deprecated, and the segmented `tocs_reduce` stays
  function-shaped permanently. Array-first consumers keep calling these
  directly, at zero wrapping cost.
- **The object is ergonomics.** `Toc` is a thin view over the canonical
  `uint64` word set — `toc_normalize`'s sorted maximal merges — never a new
  representation: **every public method is a single delegation to a kernel
  function** (all three delegate to `toc_and`, the one set operation the
  issue #177 call-site audit ruled in). The array stays the interchange
  format — `Toc.__toc_words__()` hands the canonical words back, and any
  object exposing that dunder is accepted wherever a `Toc` is.

## The canonical form is a word set

A store observed in campaigns has *gappy* coverage: one merged envelope
papers over the gaps exactly where they are most informative, so the
canonical form keeps k disjoint spans (plus free instants, bit-identical).
Normalization is **lossy toward coverage, one way**: a timestamp subsumed by
a range's decoded span is absorbed at construction, and a cover can be
rebuilt from the sibling word arrays it came from — never the arrays from a
cover. Union needs no method (construction normalizes, so
`Toc(np.append(a.words, b.words))` is the union), and the difference /
symmetric-difference directions deliberately do not ship: conservative
covers under-cover on subtraction, and no audited call site exists.

Two naming notes. `Toc.overlaps` / `Toc.contains` compare two whole covers
and answer once; the un-deprecated kernel predicates `toc_overlaps` /
`toc_contains` of the same names take a `[q_start_ns, q_end_ns)` query
window and answer elementwise, per word — a different question. And the
predicates are *envelope* algebra, not data algebra: the
conservative-direction table in the module docstring below says which way
each answer can err near a span edge (the quanta are ~2–4 s).

::: mortie.toc_object
    options:
      members:
        - Toc
