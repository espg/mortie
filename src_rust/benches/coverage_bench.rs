use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use mortie_rustie::coverage::{polygon_to_morton_coverage, polygon_to_morton_moc};

// ---------------------------------------------------------------------------
// Synthetic polygon data
// ---------------------------------------------------------------------------

/// Simple triangle ~10° × 10°, mid-latitude (well inside the equatorial zone,
/// clear of the 41.8° HEALPix transition) so it exercises the common
/// great-circle-quad straddle path.
fn triangle() -> (Vec<f64>, Vec<f64>) {
    (
        vec![20.0, 30.0, 25.0],
        vec![-120.0, -120.0, -110.0],
    )
}

/// Square ~10° × 10°, mid-latitude (common path).
fn square() -> (Vec<f64>, Vec<f64>) {
    (
        vec![20.0, 20.0, 30.0, 30.0],
        vec![-125.0, -115.0, -115.0, -125.0],
    )
}

/// Near-pole triangle whose HEALPix cell edges curve significantly: exercises
/// the densified-boundary straddle path (issue #32).  No mid-latitude twin, so
/// it establishes its own baseline rather than regressing an existing one.
fn triangle_polar() -> (Vec<f64>, Vec<f64>) {
    (
        vec![-80.0, -88.0, -84.0],
        vec![-120.0, -120.0, -100.0],
    )
}

/// Near-pole square (densified-boundary path, issue #32).
fn square_polar() -> (Vec<f64>, Vec<f64>) {
    (
        vec![-80.0, -80.0, -87.0, -87.0],
        vec![-130.0, -100.0, -100.0, -130.0],
    )
}

/// Complex polygon with ~100 vertices (circle approximation)
fn circle_polygon(n: usize) -> (Vec<f64>, Vec<f64>) {
    let center_lat = -75.0_f64;
    let center_lon = 0.0_f64;
    let radius = 5.0_f64; // degrees

    let mut lats = Vec::with_capacity(n);
    let mut lons = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        lats.push(center_lat + radius * angle.cos());
        lons.push(center_lon + radius * angle.sin());
    }
    (lats, lons)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_triangle(c: &mut Criterion) {
    let (lats, lons) = triangle();
    let mut group = c.benchmark_group("coverage_triangle");
    for order in [4u8, 6, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, &order| {
            b.iter(|| {
                polygon_to_morton_coverage(black_box(&lats), black_box(&lons), black_box(order))
            })
        });
    }
    group.finish();
}

fn bench_square(c: &mut Criterion) {
    let (lats, lons) = square();
    let mut group = c.benchmark_group("coverage_square");
    for order in [4u8, 6, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, &order| {
            b.iter(|| {
                polygon_to_morton_coverage(black_box(&lats), black_box(&lons), black_box(order))
            })
        });
    }
    group.finish();
}

fn bench_triangle_polar(c: &mut Criterion) {
    let (lats, lons) = triangle_polar();
    let mut group = c.benchmark_group("coverage_triangle_polar");
    for order in [4u8, 6, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, &order| {
            b.iter(|| {
                polygon_to_morton_coverage(black_box(&lats), black_box(&lons), black_box(order))
            })
        });
    }
    group.finish();
}

fn bench_square_polar(c: &mut Criterion) {
    let (lats, lons) = square_polar();
    let mut group = c.benchmark_group("coverage_square_polar");
    for order in [4u8, 6, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, &order| {
            b.iter(|| {
                polygon_to_morton_coverage(black_box(&lats), black_box(&lons), black_box(order))
            })
        });
    }
    group.finish();
}

fn bench_circle_polygon(c: &mut Criterion) {
    let mut group = c.benchmark_group("coverage_circle");
    for n_verts in [32usize, 100, 500] {
        let (lats, lons) = circle_polygon(n_verts);
        group.bench_with_input(
            BenchmarkId::new("order6", n_verts),
            &n_verts,
            |b, _| {
                b.iter(|| {
                    polygon_to_morton_coverage(
                        black_box(&lats),
                        black_box(&lons),
                        black_box(6),
                    )
                })
            },
        );
    }
    group.finish();
}

/// High-vertex circle across orders — the #29 regression corner (many vertices
/// at coarse order) plus deeper orders where the interior dominates.
fn bench_circle_orders(c: &mut Criterion) {
    let (lats, lons) = circle_polygon(500);
    let mut group = c.benchmark_group("coverage_circle500");
    for order in [6u8, 8, 10] {
        group.bench_with_input(BenchmarkId::from_parameter(order), &order, |b, &order| {
            b.iter(|| {
                polygon_to_morton_coverage(black_box(&lats), black_box(&lons), black_box(order))
            })
        });
    }
    group.finish();
}

/// Flat single-order output vs. compact multi-order (MOC) output, for a polygon
/// with a large interior where the MOC collapses to a few coarse cells.
fn bench_flat_vs_moc(c: &mut Criterion) {
    let (lats, lons) = circle_polygon(100);
    let mut group = c.benchmark_group("coverage_output");
    for order in [8u8, 10] {
        group.bench_with_input(BenchmarkId::new("flat", order), &order, |b, &order| {
            b.iter(|| {
                polygon_to_morton_coverage(black_box(&lats), black_box(&lons), black_box(order))
            })
        });
        group.bench_with_input(BenchmarkId::new("moc", order), &order, |b, &order| {
            b.iter(|| {
                polygon_to_morton_moc(black_box(&lats), black_box(&lons), black_box(order))
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_triangle,
    bench_square,
    bench_triangle_polar,
    bench_square_polar,
    bench_circle_polygon,
    bench_circle_orders,
    bench_flat_vs_moc
);
criterion_main!(benches);
