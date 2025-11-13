use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

// Inlined for benchmarking (cdylib can't be imported)
const POWERS_OF_10: [i64; 19] = [
    1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,
    100_000_000, 1_000_000_000, 10_000_000_000, 100_000_000_000,
    1_000_000_000_000, 10_000_000_000_000, 100_000_000_000_000,
    1_000_000_000_000_000, 10_000_000_000_000_000, 100_000_000_000_000_000,
    1_000_000_000_000_000_000,
];

const POWERS_OF_4: [i64; 19] = [
    1, 4, 16, 64, 256, 1_024, 4_096, 16_384, 65_536, 262_144,
    1_048_576, 4_194_304, 16_777_216, 67_108_864, 268_435_456,
    1_073_741_824, 4_294_967_296, 17_179_869_184, 68_719_476_736,
];

#[inline]
fn fast_norm2mort_scalar(order: i64, normed: i64, parent: i64) -> i64 {
    if order > 18 {
        panic!("Max order is 18");
    }
    let order_usize = order as usize;
    let mut mask = 3 * POWERS_OF_4[order_usize - 1];
    let mut num: i64 = 0;
    for i in (1..=order).rev() {
        let i_usize = i as usize;
        let next_bit = (normed & mask) >> ((2 * i) - 2);
        num += (next_bit + 1) * POWERS_OF_10[i_usize - 1];
        mask >>= 2;
    }
    if parent >= 6 {
        let mut parents = parent - 11;
        parents *= POWERS_OF_10[order_usize];
        num += parents;
        num = -num;
        num -= 6 * POWERS_OF_10[order_usize];
    } else {
        let parents = (parent + 1) * POWERS_OF_10[order_usize];
        num += parents;
    }
    num
}

fn bench_scalar(c: &mut Criterion) {
    c.bench_function("fast_norm2mort_scalar", |b| {
        b.iter(|| {
            fast_norm2mort_scalar(black_box(18), black_box(1000), black_box(2))
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
                    .map(|(&n, &p)| fast_norm2mort_scalar(black_box(18), black_box(n), black_box(p)))
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
                fast_norm2mort_scalar(black_box(order), black_box(1000), black_box(2))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_scalar, bench_batch, bench_different_orders);
criterion_main!(benches);
