# mortie.orders

Querying, changing and validating a packed word's HEALPix **order**, plus the
resolution ladder those orders sit on. Split out of `mortie.tools` by domain
(issue #159) so the Python surface mirrors the Rust tree (`morton.rs`); the
dense batch kernel behind `generate_morton_children`'s array form lives in
[mortie.batch](batch.md) as a private function (issues #170, #187).
The names stay flat on the package (`mortie.orders_of`, `mortie.clip2order`).

::: mortie.orders
    options:
      members:
        - infer_order_from_morton
        - orders_of
        - orders_of_uniq
        - is_point
        - validate_morton
        - clip2order
        - generate_morton_children
        - order2res
        - res2display
