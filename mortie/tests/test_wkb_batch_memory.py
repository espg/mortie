"""Peak-memory posture of the WKB batch (issue #157, phase 3 review fold).

:func:`mortie.from_wkbs` documents a peak of *result + one chunk of copied
input bytes + one chunk of in-flight covers*.  Two things make that a fact
rather than an aspiration, and neither is visible to a correctness test:

1. the input contract is screened **without materializing**, so a hex or
   buffer column is not resident a second time for the length of the call;
2. a chunk ends at a **byte budget** as well as a blob count, so a column of
   fat geometries cannot turn "one chunk" into gigabytes.

Both cases run in a fresh subprocess and read ``ru_maxrss``, which is a
high-water mark taken after the column is built -- so a build that peaks above
the call under-reports the growth and can only make these tests *pass* too
easily, never fail spuriously.  Thresholds are set with several times the
margin measured on the fixed code.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("resource", reason="ru_maxrss needs the POSIX resource module")

PREAMBLE = """
import resource, struct, sys
import numpy as np
import mortie

def maxrss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024

def quad(i):
    lon, lat = (i % 360) - 180.0, ((i * 7) % 140) - 70.0
    ring = ((lon, lat), (lon + 1, lat), (lon + 1, lat + 1),
            (lon, lat + 1), (lon, lat))
    return struct.pack("<BIII", 1, 3, 1, 5) + b"".join(
        struct.pack("<dd", x, y) for x, y in ring)

def fat(nvert=65536):
    t = np.linspace(0.0, 2.0 * np.pi, nvert)
    lon, lat = 0.01 * np.cos(t), 0.01 * np.sin(t)
    lon[-1], lat[-1] = lon[0], lat[0]
    return (struct.pack("<BIII", 1, 3, 1, nvert)
            + np.column_stack([lon, lat]).astype("<f8").tobytes())
"""


def growth_mib(body, threads=2):
    """Run *body* in a fresh interpreter; return its reported RSS growth in MiB.

    The rayon pool is pinned small so the *in-flight cover* term stays a
    couple of blobs' worth and the measurement is dominated by the input copy,
    which is the term under test.

    Parameters
    ----------
    body : str
        Python source binding ``blobs`` and ``ORDER``.
    threads : int, optional
        ``RAYON_NUM_THREADS`` for the child.  Default 2.

    Returns
    -------
    float
        Peak RSS growth over the post-build baseline, in MiB.
    """
    script = PREAMBLE + textwrap.dedent(body) + textwrap.dedent("""
        base = maxrss_mib()
        mortie.from_wkbs(blobs, order=ORDER)
        print(maxrss_mib() - base)
        """)
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, check=True, env=env)
    return float(out.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_a_hex_column_costs_no_more_peak_than_a_bytes_column():
    # The pre-pass used to hold a coerced `bytes` for every non-`bytes` entry
    # for the whole call, which put the column in memory twice: measured at
    # 2.7x the result on the ATL03 corpus against 1.07x for `bytes`.  Built
    # one blob at a time so the *build* never holds both spellings at once.
    as_bytes = growth_mib("""
        blobs = [quad(i) for i in range(300_000)]
        ORDER = 5
        """)
    as_hex = growth_mib("""
        blobs = [quad(i).hex() for i in range(300_000)]
        ORDER = 5
        """)
    # The column is ~27 MiB of WKB; doubling it is unmissable at this margin.
    assert as_hex < as_bytes + 15.0, f"bytes={as_bytes:.1f} hex={as_hex:.1f} MiB"


@pytest.mark.slow
def test_the_chunk_copy_is_capped_in_bytes_not_only_in_blob_count():
    # 200 x ~1 MiB is one chunk by blob count (< 2048), so without the byte
    # budget the whole 200 MiB column is copied into one buffer.  The blobs are
    # one shared object, so the column is not resident 200 times and what is
    # measured is the copy plus the in-flight covers.  Measured either side of
    # the budget on this machine: 319 MiB uncapped, 153-186 MiB capped -- hence
    # a threshold between them with margin on both sides, not a tight bound
    # (the residue is the in-flight cover term, which is allocator-dependent).
    growth = growth_mib("""
        blobs = [fat()] * 200
        ORDER = 5
        """)
    assert growth < 250.0, f"peak growth {growth:.1f} MiB"


BASIN_COORDS = Path("mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt")

BASIN_COLUMN = f"""
    BASIN_COORDS = r"{BASIN_COORDS}"
    import numpy as np
    table = np.loadtxt(BASIN_COORDS)
    basins = []
    for b in np.unique(table[:, 2]).astype(int):
        m = table[:, 2] == b
        la, lo = table[m, 0], table[m, 1]
        if la[0] != la[-1] or lo[0] != lo[-1]:
            la, lo = np.append(la, la[0]), np.append(lo, lo[0])
        xy = np.empty(la.size * 2)
        xy[0::2], xy[1::2] = lo, la
        basins.append(struct.pack("<BIII", 1, 3, 1, la.size)
                      + xy.astype("<f8").tobytes())
    del table
    blobs = basins * 22          # 594 blobs, a 416 MiB column
    ORDER = 6
"""


@pytest.mark.slow
def test_the_byte_cap_holds_on_the_real_antarctic_basins():
    # The synthetic case above proves the cap; this one proves it on the
    # fixture class it was added for -- the in-tree Antarctic basins, 0.65 MiB
    # median and 1.25 MiB max, repeated to a 416 MiB column.  The blobs are
    # 27 shared objects, so the column is not resident 22 times over and what
    # is measured is the chunk copy plus the in-flight covers.
    #
    # Uncapped, one chunk would be the whole column: the copy alone would be
    # 416 MiB.  Capped at 64 MiB it measures 127-254 MiB across runs (the
    # residue is the cover work, which is bounded by thread count, not by the
    # chunk), so the threshold sits above the worst observed run and well
    # below what an uncapped copy could not avoid paying.
    if not BASIN_COORDS.exists():
        pytest.skip("Antarctic polygon data not found")
    growth = growth_mib(BASIN_COLUMN)
    assert growth < 350.0, f"peak growth {growth:.1f} MiB"
