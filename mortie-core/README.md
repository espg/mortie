# mortie-core

Packed-word morton codec for HEALPix grids — the id codec of
[mortie](https://github.com/espg/mortie), split out as a dependency-free crate
so external composers (first consumer: healpix-geo) can encode and decode
mortie words without the Python package or its pyo3 surface.

A morton index here is **one unsigned 64-bit word** carrying a HEALPix NESTED
cell (or an order-29 point) *with its order encoded intrinsically*, such that
the plain unsigned sort of words is a Z-order traversal. The crate provides
the packed-word grammar and its encode/decode, order/truncation/containment
arithmetic, the decimal-string grammar, and the `(depth, nested-ipix) ↔
packed-word` pivot primitives:

```rust
use mortie_core::{from_nested, to_nested};

// HEALPix NESTED cell 1234 at order 6 <-> one packed u64 word.
let word = from_nested(1234, 6);
assert_eq!(to_nested(word), Some((6, 1234)));
```

## Stability: unstable API, frozen wire format

Two different contracts, deliberately decoupled:

- **The crate API is 0.x and unstable.** Signatures and module layout may
  change on minor bumps until the API has survived first external contact;
  `mortie-core` versions independently of the mortie Python package (which is
  tag-synced to PyPI), and its 1.0.0 is gated on the healpix-geo dependency
  landing.
- **The byte-level codec grammar is frozen for the mortie 1.x series.** What
  a word *means* — the
  [packed 64-bit morton word](https://espg.github.io/mortie/latest/specification/#1-the-packed-64-bit-morton-word)
  bit layout and the
  [decimal string grammar](https://espg.github.io/mortie/latest/specification/#2-decimal-string-representation)
  — is normative in the
  [mortie specification](https://espg.github.io/mortie/latest/specification/)
  ([`docs/specification.md`](https://github.com/espg/mortie/blob/main/docs/specification.md)
  in the repo), not in this crate's version number. Words you encode today
  stay decodable by every 1.x-era reader; that page's "Frozen for 1.x"
  section is the authoritative list of what cannot change.

## Zero dependencies

The crate is contractually dependency-minimal: no non-std dependencies, and
never a moc-crate dependency (set-ops/RangeMOC layers belong elsewhere).
`cargo tree -p mortie-core` is one line, and `tests/dep_contract.rs` fails
`cargo test` if any dependency table appears in the manifest. It also builds
for `wasm32-unknown-unknown`, matching its wasm-targeting consumers.

## MSRV

Rust **1.67** (`u64::ilog10`). Checked in CI against the packaged crate;
bumps are minor-version events while the crate is 0.x.

## License

MIT, like the rest of the mortie repository; the license text ships inside
the published crate archive. That makes the intended dependency direction
clean: an Apache-2.0 project (e.g. healpix-geo) can depend on this MIT crate
with attribution preserved and no copyleft obligations in either direction.
