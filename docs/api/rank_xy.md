# mortie.rank_xy

Subtree-local rank <-> face-local `(x, y)` bit deinterleave for 2-D block
views (issue #149). A depth-`d` subtree holds `4**d` cells whose ascending
packed-word order is a Z-order (morton) curve over a `2**d x 2**d` block;
`rank_to_xy` / `xy_to_rank` convert between a cell's **rank** in that block
and the deinterleaved pair, matching the healpy / HEALPix C++ `pix2xyf`
convention (origin at the subtree's south corner). The input is rank-space,
**not** packed morton words — strip the shard prefix down to the base-4
digit-tail rank first. Normative statement:
[specification.md §8](../specification.md#8-rank-space-x-y-deinterleave);
the public functions ship the Rust kernel (`src_rust/src/rank_xy.rs`). The
names stay flat on the package (`mortie.rank_to_xy`, `mortie.xy_to_rank`).

::: mortie.rank_xy
    options:
      members:
        - rank_to_xy
        - xy_to_rank
