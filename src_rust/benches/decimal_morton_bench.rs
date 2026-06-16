//! Benchmarks comparing the new full-resolution `decimal_morton` 64-bit kernel
//! (issue #35) against the existing order-<=18 decimal-Morton path, for the
//! operations @espg asked about on PR #43: **encoding and decoding lat+lon
//! tuples** at orders the old path supports (6, 12, 17).
//!
//! Two pipelines are timed on the *same* inputs:
//!
//! * **old** — `geo2mort_scalar` (healpix hash -> `fast_norm2mort_scalar`,
//!   decimal-packed i64) and its inverse `mort2nested`. Capped at order 18.
//! * **new** — healpix hash -> per-order 2-bit tuples -> `decimal_morton::encode`
//!   (packed 64-bit MOC word) and its inverse `decimal_morton::decode`.
//!
//! The healpix `hash` is shared by both encoders, so the *delta* the benchmark
//! isolates is the bit-packing layer (decimal string-digit math vs. the packed
//! prefix/body/suffix word), which is the part issue #35 replaces. To keep that
//! delta honest, the new encode path decomposes the nested index into a
//! **stack** tuple buffer (`[u8; 18]`, no per-point heap allocation) before
//! calling `encode`, matching `geo2mort_scalar`'s alloc-free profile — the
//! decimal_morton `decode` *does* return an owned `Vec<u8>` (its own API), so the
//! decode side carries that allocation on the new side only, noted there.
//!
//! Coverage/polygon code is unaffected by this PR — nothing calls
//! `decimal_morton` yet — so its benchmarks live unchanged in
//! `coverage_bench.rs`; this file does not duplicate them.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use healpix::coords::Degrees;
use healpix::get;

use mortie_rustie::decimal_morton;
use mortie_rustie::geo2mort::geo2mort_scalar;
use mortie_rustie::morton::mort2nested;

// ---------------------------------------------------------------------------
// Shared input data
// ---------------------------------------------------------------------------

/// A deterministic spread of mid-latitude (lat, lon) pairs covering both
/// hemispheres and a range of base cells, so neither encoder hits a trivial
/// constant-fold.
fn sample_points(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let f = i as f64;
            let lat = -70.0 + (f * 7.3) % 140.0;
            let lon = -180.0 + (f * 13.7) % 360.0;
            (lat, lon)
        })
        .collect()
}

/// Decompose a HEALPix NESTED index at `order` into the `decimal_morton`
/// inputs: the base cell and the per-order stored `0..=3` tuples written into a
/// caller-provided `buf` (order 1 is the most significant 2-bit pair below the
/// base). Returns the base cell; the first `order` entries of `buf` are filled.
/// A stack buffer keeps the new encode path allocation-free, so the comparison
/// against the alloc-free `geo2mort_scalar` isolates the bit-packing layer
/// rather than a per-point `Vec` malloc (orders here are <=18, hence `[u8; 18]`).
#[inline]
fn nested_to_tuples(nested: u64, order: u8, buf: &mut [u8; 18]) -> u8 {
    let base = (nested >> (2 * order as u32)) as u8;
    for n in 1..=order {
        let shift = 2 * (order - n) as u32;
        buf[(n - 1) as usize] = ((nested >> shift) & 3) as u8;
    }
    base
}

// ---------------------------------------------------------------------------
// Encode: lat/lon -> index
// ---------------------------------------------------------------------------

fn bench_encode(c: &mut Criterion) {
    let pts = sample_points(10_000);
    let mut group = c.benchmark_group("encode_latlon");

    for order in [6u8, 12, 17] {
        // old: geo2mort (healpix hash + decimal-pack)
        group.bench_with_input(BenchmarkId::new("old_geo2mort", order), &order, |b, &o| {
            b.iter(|| {
                let mut acc = 0i64;
                for &(lat, lon) in &pts {
                    acc ^= geo2mort_scalar(black_box(lat), black_box(lon), o);
                }
                acc
            })
        });

        // new: healpix hash + decimal_morton::encode (packed word)
        group.bench_with_input(
            BenchmarkId::new("new_decimal_morton", order),
            &order,
            |b, &o| {
                let layer = get(o);
                b.iter(|| {
                    let mut acc = 0u64;
                    let mut buf = [0u8; 18];
                    for &(lat, lon) in &pts {
                        let nested = layer.hash(Degrees(black_box(lon), black_box(lat)));
                        let base = nested_to_tuples(nested, o, &mut buf);
                        acc ^= decimal_morton::encode(base, &buf[..o as usize], o);
                    }
                    acc
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Decode: index -> components
// ---------------------------------------------------------------------------

fn bench_decode(c: &mut Criterion) {
    let pts = sample_points(10_000);
    let mut group = c.benchmark_group("decode_index");

    for order in [6u8, 12, 17] {
        // Pre-encode both representations so the loop times only the decode.
        let layer = get(order);
        let old_words: Vec<i64> = pts
            .iter()
            .map(|&(lat, lon)| geo2mort_scalar(lat, lon, order))
            .collect();
        let new_words: Vec<u64> = pts
            .iter()
            .map(|&(lat, lon)| {
                let nested = layer.hash(Degrees(lon, lat));
                let mut buf = [0u8; 18];
                let base = nested_to_tuples(nested, order, &mut buf);
                decimal_morton::encode(base, &buf[..order as usize], order)
            })
            .collect();

        // old: mort2nested (decimal-digit unpack -> nested + depth)
        group.bench_with_input(
            BenchmarkId::new("old_mort2nested", order),
            &order,
            |b, _| {
                b.iter(|| {
                    let mut acc = 0u64;
                    for &w in &old_words {
                        let (nested, depth) = mort2nested(black_box(w));
                        acc ^= nested ^ depth as u64;
                    }
                    acc
                })
            },
        );

        // new: decimal_morton::decode (packed-word unpack). NB: decode returns an
        // owned `Vec<u8>` of tuples (its API), so the new side alone pays a
        // per-word allocation here that `mort2nested` (scalar return) does not.
        group.bench_with_input(
            BenchmarkId::new("new_decimal_morton", order),
            &order,
            |b, _| {
                b.iter(|| {
                    let mut acc = 0u64;
                    for &w in &new_words {
                        let dec = decimal_morton::decode(black_box(w)).expect("decode");
                        acc ^= dec.base_cell as u64 ^ dec.order as u64;
                    }
                    acc
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
