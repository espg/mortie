# mortie.buffer

Cell-set dilation: morton indices in, morton indices out. Split out of
`mortie.tools` by domain (issue #159) so the Python surface mirrors the Rust
tree (`buffer.rs`); the names stay flat on the package
(`mortie.morton_buffer`).

::: mortie.buffer
    options:
      members:
        - morton_buffer
        - morton_buffer_meters
