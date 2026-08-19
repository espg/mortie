"""Verify WKB ingest against a real corpus, per issue #157's acceptance.

The acceptance criterion is byte parity **with the shapely-backed path** on
real data — the 555,867-granule ATL03 v007 catalog was the reference corpus —
and the fixture-scale half of that lives in the test suite
(``test_wkb_reader.py``, ``test_geometry.py``, ``test_wkb_basins.py``).  The
catalog-scale half cannot: the column is ~300 MB and lives outside the repo.
This script is that half, made re-runnable by anyone holding such a column
rather than quoted from a transcript.

It checks two things per blob, which are different claims:

* ``from_wkb`` (Rust reader) == ``from_geometry(shapely.from_wkb(blob))`` —
  parity with the path the reader replaced.  Skipped without shapely.
* ``from_wkbs`` batch entry ``i`` == ``from_wkb`` on blob ``i`` — the batch's
  identity contract.

Run:
    python benchmarks/verify_wkb_corpus_parity.py COLUMN.parquet [order] \\
        [--column geometry] [--sample N]

``--sample N`` checks a deterministic N-blob subset (the batch still runs over
the whole column, so the ragged contract is exercised in full); omit it for
the complete pass.  Exit status is non-zero if any blob disagrees.
"""

import argparse
import sys
import time

import numpy as np

import mortie
from mortie.batch import _from_wkbs


def load_column(path, name):
    """Read a binary/WKB column from a parquet file as a list of blobs.

    Parameters
    ----------
    path : str
        The parquet file to read.
    name : str
        The column holding WKB bytes.

    Returns
    -------
    list of bytes
        One blob per row.
    """
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=[name])[name].to_pylist()


def main():
    """Run the corpus parity checks and report the results.

    Returns
    -------
    int
        ``0`` when every checked blob agrees, ``1`` otherwise (the process
        exit status).
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet")
    ap.add_argument("order", nargs="?", type=int, default=6)
    ap.add_argument("--column", default="geometry")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    blobs = load_column(args.parquet, args.column)
    n = len(blobs)
    payload = sum(len(b) for b in blobs) / 1e6
    print(f"n={n} payload={payload:.1f} MB order={args.order}", flush=True)

    t0 = time.perf_counter()
    values, offsets = _from_wkbs(blobs, order=args.order)
    t_batch = time.perf_counter() - t0
    print(f"from_wkbs: {t_batch:.2f} s ({1e6 * t_batch / n:.1f} us/blob), "
          f"{values.size} cells", flush=True)

    assert offsets[0] == 0, "offsets must start at 0"
    assert offsets[-1] == values.size, "offsets must end at len(values)"
    assert offsets.size == n + 1, "one offset per blob, plus the endpoint"
    print("ragged contract OK", flush=True)

    try:
        import shapely
    except ImportError:
        shapely = None
        print("shapely absent — skipping the backend-path comparison", flush=True)

    idx = range(n) if not args.sample else np.linspace(
        0, n - 1, min(args.sample, n), dtype=int
    )
    bad_scalar = bad_backend = 0
    t0 = time.perf_counter()
    for k, i in enumerate(idx):
        scalar = mortie.from_wkb(blobs[i], order=args.order, moc=True)
        if not np.array_equal(values[offsets[i]:offsets[i + 1]], scalar):
            bad_scalar += 1
            print(f"  BATCH MISMATCH at {i}", flush=True)
        if shapely is not None:
            backend = mortie.from_geometry(
                shapely.from_wkb(blobs[i]), order=args.order, moc=True
            )
            if not np.array_equal(scalar, backend):
                bad_backend += 1
                print(f"  BACKEND MISMATCH at {i}", flush=True)
        if k and k % 100_000 == 0:
            print(f"  ...{k} checked, {time.perf_counter() - t0:.0f}s", flush=True)
    checked = len(idx) if args.sample else n
    print(f"checked {checked} blobs in {time.perf_counter() - t0:.2f} s", flush=True)
    print(f"batch   vs scalar : {checked - bad_scalar}/{checked} identical", flush=True)
    if shapely is not None:
        print(f"reader  vs shapely: {checked - bad_backend}/{checked} identical",
              flush=True)
    return 1 if (bad_scalar or bad_backend) else 0


if __name__ == "__main__":
    sys.exit(main())
