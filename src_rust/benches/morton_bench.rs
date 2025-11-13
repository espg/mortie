use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// Re-export the morton module for benchmarking
// In a real project, you'd import from the crate
mod morton {
    include!("../src/morton.rs");
}

fn bench_scalar(c: &mut Criterion) {
    c.bench_function("fast_norm2mort_scalar", |b| {
        b.iter(|| {
            morton::fast_norm2mort_scalar(black_box(18), black_box(1000), black_box(2))
        });
    });
}

fn bench_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_norm2mort_batch");

    for size in [100, 1_000, 10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let normed: Vec<i64> = (0..size).collect();
            let parents: Vec<i64> = (0..size).map(|i| i % 12).collect();

            b.iter(|| {
                let _results: Vec<i64> = normed
                    .iter()
                    .zip(parents.iter())
                    .map(|(&n, &p)| morton::fast_norm2mort_scalar(black_box(18), black_box(n), black_box(p)))
                    .collect();
            });
        });
    }
    group.finish();
}

fn bench_different_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_norm2mort_orders");

    for order in [6, 10, 14, 18].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(order), order, |b, &order| {
            b.iter(|| {
                morton::fast_norm2mort_scalar(black_box(order), black_box(1000), black_box(2))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_scalar, bench_batch, bench_different_orders);
criterion_main!(benches);
