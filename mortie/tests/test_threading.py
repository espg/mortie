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
