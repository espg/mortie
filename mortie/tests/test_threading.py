"""Concurrency tests for the Rust bindings.

The Rust `#[pyfunction]`s release the GIL around their pure-Rust (often
rayon-parallel) compute via `py.allow_threads`.  These tests exercise that
path from many Python threads at once and assert the results are identical to
a single-threaded run -- i.e. releasing the GIL does not corrupt output or
crash.  They are deliberately *correctness* tests, not timing tests, so they
are not flaky under CI load.
"""

import threading

import numpy as np

import mortie


def _run_concurrent(fn, n_threads=8):
    """Call ``fn`` from ``n_threads`` threads; return the list of results."""
    results = [None] * n_threads
    barrier = threading.Barrier(n_threads)

    def worker(i):
        # timeout so a worker that dies before the barrier fails the test red
        # instead of deadlocking the whole suite
        barrier.wait(timeout=30)
        results[i] = fn()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_geo2mort_concurrent():
    lats = np.linspace(-80.0, 80.0, 5000)
    lons = np.linspace(-179.0, 179.0, 5000)
    expected = mortie.geo2mort(lats, lons, order=12)
    for got in _run_concurrent(lambda: mortie.geo2mort(lats, lons, order=12)):
        np.testing.assert_array_equal(got, expected)


def test_morton_coverage_concurrent():
    lats = np.array([40.0, 42.0, 42.0, 40.0])
    lons = np.array([46.0, 46.0, 48.0, 48.0])
    expected = mortie.morton_coverage(lats, lons, order=10)
    for got in _run_concurrent(lambda: mortie.morton_coverage(lats, lons, order=10)):
        np.testing.assert_array_equal(got, expected)


def test_morton_coverage_moc_concurrent():
    # exercises the MOC path, whose binding also touches Python `warnings`
    # outside the GIL-released region
    lats = np.array([40.0, 42.0, 42.0, 40.0])
    lons = np.array([46.0, 46.0, 48.0, 48.0])
    expected = mortie.morton_coverage_moc(lats, lons, order=12)
    for got in _run_concurrent(lambda: mortie.morton_coverage_moc(lats, lons, order=12)):
        np.testing.assert_array_equal(got, expected)


def test_morton_buffer_concurrent():
    cells = mortie.morton_coverage(
        np.array([40.0, 42.0, 42.0, 40.0]),
        np.array([46.0, 46.0, 48.0, 48.0]),
        order=10,
    )
    expected = mortie.morton_buffer(cells, k=1)
    for got in _run_concurrent(lambda: mortie.morton_buffer(cells, k=1)):
        np.testing.assert_array_equal(got, expected)


def test_mort2geo_concurrent():
    # decode path: exercises rust_pix2ang / rust_vec2ang under concurrency
    cells = mortie.morton_coverage(
        np.array([40.0, 42.0, 42.0, 40.0]),
        np.array([46.0, 46.0, 48.0, 48.0]),
        order=10,
    )
    exp_lat, exp_lon = mortie.mort2geo(cells)
    for got_lat, got_lon in _run_concurrent(lambda: mortie.mort2geo(cells)):
        np.testing.assert_array_equal(got_lat, exp_lat)
        np.testing.assert_array_equal(got_lon, exp_lon)


def test_linestring_coverage_concurrent():
    lats = np.array([40.0, 41.0, 42.0])
    lons = np.array([46.0, 47.0, 48.0])

    def call():
        return mortie.linestring_coverage(lats, lons, order=10)

    expected = call()
    for got in _run_concurrent(call):
        np.testing.assert_array_equal(got, expected)


def test_gil_released_during_rust_compute():
    """Prove the GIL is actually released during compute, not merely that output
    is uncorrupted (the other tests in this file pass even on unmodified ``main``).

    A pure-Python counter thread spins while a heavy Rust call runs.  With the
    GIL held for the whole C call (no ``allow_threads``), CPython cannot preempt
    the call, so the counter cannot advance during it; with the GIL released, the
    Python thread runs freely and the counter climbs by many thousands.
    """
    lats = np.linspace(-80.0, 80.0, 2_000_000)
    lons = np.linspace(-179.0, 179.0, 2_000_000)

    counter = [0]
    stop = threading.Event()
    started = threading.Event()

    def busy():
        started.set()
        while not stop.is_set():
            counter[0] += 1

    b = threading.Thread(target=busy)
    b.start()
    started.wait()
    try:
        # Bracket the GIL-released region: read the counter immediately before
        # and after the Rust call from this thread.
        pre = counter[0]
        mortie.geo2mort(lats, lons, order=12)
        progressed = counter[0] - pre
    finally:
        stop.set()
        b.join()

    # GIL held for the whole call -> progressed ~= 0; released -> >> 1000.
    assert progressed > 1000, (
        f"Python thread made ~no progress during the Rust compute ({progressed}); "
        "the GIL was likely held (allow_threads not in effect)"
    )
