"""Peak-memory posture of the WKB batch (issue #157, phase 3 review fold).

:func:`mortie.from_wkbs` documents a peak of *result + one chunk of copied
input bytes + one chunk of in-flight covers*.  What makes that a fact rather
than an aspiration is invisible to a correctness test: a chunk ends at a
**byte budget** as well as a blob count, so a column of fat geometries cannot
turn "one chunk" into gigabytes.

The case runs in a fresh subprocess and read ``ru_maxrss``, which is a
high-water mark taken after the column is built -- so a build that peaks above
the call under-reports the growth and can only make these tests *pass* too
easily, never fail spuriously.  Thresholds are set with several times the
margin measured on the fixed code.
"""

import os
import subprocess
import sys
import textwrap

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
