use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use mortie_rustie::coverage::polygon_to_morton_coverage;

// ---------------------------------------------------------------------------
// Synthetic polygon data
// ---------------------------------------------------------------------------

/// Simple triangle ~10° × 10°
fn triangle() -> (Vec<f64>, Vec<f64>) {
    (
        vec![40.0, 50.0, 45.0],
        vec![-120.0, -120.0, -110.0],
    )
}

/// Square ~10° × 10°
fn square() -> (Vec<f64>, Vec<f64>) {
    (
        vec![40.0, 40.0, 50.0, 50.0],
        vec![-125.0, -115.0, -115.0, -125.0],
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

criterion_group!(benches, bench_triangle, bench_square, bench_circle_polygon);
criterion_main!(benches);
