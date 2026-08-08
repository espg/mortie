"""Peak-memory posture of the WKB batch (issue #157, phase 3 review fold).

:func:`mortie.from_wkbs` documents a peak of *result + one chunk of copied
input bytes + one chunk of in-flight covers*.  Two things make that a fact
rather than an aspiration, and neither is visible to a correctness test:

1. the input contract is screened **without materializing**, so a hex or
   buffer column is not resident a second time for the length of the call;
2. a chunk ends at a **byte budget** as well as a blob count, so a column of
   fat geometries cannot turn "one chunk" into gigabytes.

Both cases run in a fresh subprocess.  The **baseline is resident RSS** once
the column is built, and the **peak** is ``ru_maxrss``; growth is the
difference.  The baseline deliberately is *not* ``ru_maxrss``, because that is
a process-lifetime high-water: the basin column below is built with
``np.loadtxt`` over a 1.24M-line fixture, whose transient sits a
run-dependent 0-28 MiB above what is actually resident when the call starts,
and taking it as the baseline subtracts that transient from the growth --
under-reporting by as much as the term under test is worth on a small column.

What is left is conservative in the safe direction: the peak is still a
lifetime high-water, so if a build transient ever exceeded the call's own peak
the growth would be **over**-reported, which can only make a threshold fail,
never pass wrongly.  Growth is also a single sample, so quote it as a range
over repeated runs rather than as a point estimate; thresholds here are set
with several times the margin measured on the fixed code.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("resource", reason="ru_maxrss needs the POSIX resource module")

PREAMBLE = """
import gc, os, resource, struct, subprocess, sys
import numpy as np
import mortie

def maxrss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024

def resident_mib():
    try:
        with open("/proc/self/statm") as fh:            # Linux
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except OSError:                                     # macOS has no /proc
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip()) / 1024

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


def growth_samples(body, threads=2, reps=3):
    """Run *body* in *reps* fresh interpreters; return each one's RSS growth.

    The baseline is the child's **resident** RSS once the column is built and
    collected; the peak is ``ru_maxrss`` at the end of the call (see the module
    docstring for why the two differ and which way the residue errs).  The
    rayon pool is pinned small so the *in-flight cover* term stays a couple of
    blobs' worth and the measurement is dominated by the input copy, which is
    the term under test.

    Parameters
    ----------
    body : str
        Python source binding ``blobs`` and ``ORDER``.
    threads : int, optional
        ``RAYON_NUM_THREADS`` for the child.  Default 2.
    reps : int, optional
        How many child processes to run.  Default 3.

    Returns
    -------
    list of float
        One peak-growth sample per run, in MiB, in run order.
    """
    script = PREAMBLE + textwrap.dedent(body) + textwrap.dedent("""
        gc.collect()
        base = resident_mib()
        mortie.from_wkbs(blobs, order=ORDER)
        print(maxrss_mib() - base)
        """)
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads)}
    samples = []
    for _ in range(reps):
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, check=True, env=env)
        samples.append(float(out.stdout.strip().splitlines()[-1]))
    return samples


def growth_mib(body, threads=2, reps=3):
    """Peak RSS growth of *body*, in MiB: the **smallest** of *reps* runs.

    A single run is one sample of a quantity with a long right tail -- the
    system, the allocator and the rayon pool all add to it, and on this
    fixture the spread across runs is comparable to the term being measured.
    The minimum is the least contaminated estimate of the intrinsic peak, and
    it is the honest statistic for an upper-bound assertion: an *uncapped*
    copy would exceed these thresholds on every run, not merely on the worst.
    Quote a range from :func:`growth_samples` rather than this number alone.

    Parameters
    ----------
    body : str
        Python source binding ``blobs`` and ``ORDER``.
    threads : int, optional
        ``RAYON_NUM_THREADS`` for the child.  Default 2.
    reps : int, optional
        How many child processes to run.  Default 3.

    Returns
    -------
    float
        The minimum peak RSS growth over the resident post-build baseline.
    """
    return min(growth_samples(body, threads=threads, reps=reps))


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
    # the budget on this machine: >=319 MiB uncapped (that figure predates the
    # resident-baseline fix, which only ever raises a growth, so it is a lower
    # bound), against 146-177 MiB over three capped runs -- hence a threshold
    # between them with margin on both sides, not a tight bound (the residue is
    # the in-flight cover term, which is allocator-dependent).
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
    # 416 MiB.  Capped at 64 MiB, three runs measured 243/246/585 MiB -- the
    # residue is the cover work, bounded by thread count rather than by the
    # chunk, and its right tail is why `growth_mib` takes the minimum: a single
    # sample of this quantity is not reproducible (a 585 MiB run and a 243 MiB
    # run are the same code on the same column).  The threshold sits above the
    # minima observed and well below what an uncapped copy could not avoid.
    if not BASIN_COORDS.exists():
        pytest.skip("Antarctic polygon data not found")
    growth = growth_mib(BASIN_COLUMN)
    assert growth < 350.0, f"peak growth {growth:.1f} MiB"
