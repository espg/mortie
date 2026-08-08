"""Differentially validate mortie's WKB reader against published crates (#157).

espg's phase-5 direction was to implement the reader first and *then* compare
against `wkb` / `geozero` / `geo-types`.  The valuable half of that is
correctness: a mature, widely used WKB implementation is a far better source of
truth than hand-written expectations, so this script uses two of them as
**oracles** and reports every disagreement, with its direction.

Two hard constraints shape the design:

* **Nothing may leak into the shipped crate.**  The harness is generated into a
  temporary directory *outside* the repository, so `Cargo.toml` is never
  touched and the oracle crates cannot appear in mortie's dependency graph.
  Verify with ``cargo tree | grep -E 'wkb|geozero|geo-types'`` — it must be
  empty, as must ``git diff origin/main -- Cargo.toml``.
* **The parser under test must be the real one.**  ``src_rust/src/wkb.rs`` is
  copied verbatim, with only its ``pub mod batch;`` line dropped (that submodule
  pulls in rayon and the coverage kernels, which are not what is being
  compared).  The copy is made fresh on every run, so it cannot drift.

Run::

    python benchmarks/compare_wkb_crates.py [--corpus COLUMN.parquet]
                                            [--repeats 7] [--keep DIR]

Without ``--corpus`` it runs the edge fixtures and the fuzz corpus only.
"""

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BASINS = REPO / "mortie/tests/Ant_Grounded_DrainageSystem_Polygons.txt"

CARGO_TOML = """\
[package]
name = "mortie-wkb-diff"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
wkb = "0.9"
geo-traits = "0.3"
geo-types = "0.7"
geozero = { version = "0.15", default-features = false, features = [
    "with-geo", "with-wkb",
] }

[profile.release]
opt-level = 3
"""

MAIN_RS = r'''
mod mortie_wkb;

use std::env;
use std::fs;

use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geozero::ToGeo;

/// Every ring of every part as (x, y) = (lon, lat) — mortie's ring contract.
type Rings = Vec<Vec<(f64, f64)>>;

fn read_blobs(path: &str) -> Vec<Vec<u8>> {
    let raw = fs::read(path).expect("blob file");
    let (mut out, mut pos) = (Vec::new(), 0usize);
    while pos + 4 <= raw.len() {
        let n = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        out.push(raw[pos..(pos + n).min(raw.len())].to_vec());
        pos += n;
    }
    out
}

fn geo_rings(g: &Geometry<f64>) -> Result<Rings, String> {
    let ls = |l: &geo_types::LineString<f64>| l.0.iter().map(|c| (c.x, c.y)).collect();
    let poly = |p: &geo_types::Polygon<f64>| {
        let mut r: Rings = vec![ls(p.exterior())];
        r.extend(p.interiors().iter().map(ls));
        r
    };
    match g {
        Geometry::Polygon(p) => Ok(poly(p)),
        Geometry::MultiPolygon(m) => Ok(m.0.iter().flat_map(poly).collect()),
        Geometry::LineString(l) => Ok(vec![ls(l)]),
        Geometry::MultiLineString(m) => Ok(m.0.iter().map(ls).collect()),
        other => Err(format!("unsupported geo type {other:?}")),
    }
}

fn mortie_rings(bytes: &[u8]) -> Result<Rings, String> {
    let r = mortie_wkb::parse(bytes)?;
    Ok((0..r.offsets.len() - 1)
        .map(|i| {
            let (s, e) = (r.offsets[i] as usize, r.offsets[i + 1] as usize);
            (s..e).map(|k| (r.lons[k], r.lats[k])).collect()
        })
        .collect())
}

/// Bit-exact: an axis swap or a lost ulp both surface here.
fn same(a: &Rings, b: &Rings) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(ra, rb)| {
            ra.len() == rb.len()
                && ra.iter().zip(rb).all(|(p, q)| {
                    p.0.to_bits() == q.0.to_bits() && p.1.to_bits() == q.1.to_bits()
                })
        })
}

fn wkb_crate(bytes: &[u8]) -> Result<Rings, String> {
    geo_rings(&wkb::reader::read_wkb(bytes).map_err(|e| e.to_string())?.to_geometry())
}

/// geozero makes the caller pick the dialect up front — plain `Wkb` refuses an
/// SRID-prefixed blob — so try EWKB then WKB, the closest equivalent to
/// mortie's single entry point.
fn geozero_crate(bytes: &[u8]) -> Result<Rings, String> {
    let g = geozero::wkb::Ewkb(bytes)
        .to_geo()
        .or_else(|_| geozero::wkb::Wkb(bytes).to_geo())
        .map_err(|e| e.to_string())?;
    geo_rings(&g)
}

fn geozero_plain(bytes: &[u8]) -> Result<Rings, String> {
    geo_rings(&geozero::wkb::Wkb(bytes).to_geo().map_err(|e| e.to_string())?)
}

#[derive(Default)]
struct Tally {
    agree_ok: usize,
    agree_err: usize,
    agree_unsupported_type: usize,
    mortie_stricter: usize,
    oracle_stricter: usize,
    value_divergence: usize,
    causes: std::collections::BTreeMap<String, usize>,
    examples: Vec<String>,
    cap: usize,
}

impl Tally {
    fn note(&mut self, m: String) {
        if self.examples.len() < self.cap {
            self.examples.push(m);
        }
    }

    fn compare(&mut self, i: usize, name: &str, mine: &Result<Rings, String>,
               theirs: &Result<Rings, String>) {
        match (mine, theirs) {
            (Ok(a), Ok(b)) if same(a, b) => self.agree_ok += 1,
            (Ok(a), Ok(b)) => {
                self.value_divergence += 1;
                self.note(format!(
                    "[{i}] {name}: VALUES differ ({} vs {} rings)", a.len(), b.len()));
            }
            (Err(_), Err(o)) if o.starts_with("unsupported geo type") => {
                self.agree_unsupported_type += 1
            }
            (Err(_), Err(_)) => self.agree_err += 1,
            (Err(e), Ok(_)) => {
                self.mortie_stricter += 1;
                let cause = if e.contains("is not closed") { "unclosed ring" }
                    else if e.contains("byte-order flag") { "bad byte-order flag" }
                    else if e.starts_with("truncated") { "truncated" }
                    else if e.starts_with("unsupported") { "unsupported geometry type" }
                    else if e.starts_with("empty") { "empty geometry" }
                    else if e.contains("unrecognized") { "unrecognized type code" }
                    else if e.contains("expected a") { "wrong part type" }
                    else { "other" };
                *self.causes.entry(cause.to_string()).or_default() += 1;
                self.note(
                    format!("[{i}] {name}: mortie rejects ({e}), oracle accepts"));
            }
            (Ok(_), Err(o)) => {
                self.oracle_stricter += 1;
                self.note(
                    format!("[{i}] {name}: mortie accepts, oracle rejects ({o})"));
            }
        }
    }

    fn report(&self, label: &str, n: usize) {
        println!("{label:>14}: agree_ok={} agree_err={} agree_unsupported_type={} \
                  mortie_stricter={} oracle_stricter={} VALUE_DIVERGENCE={} (of {n})",
                 self.agree_ok, self.agree_err, self.agree_unsupported_type,
                 self.mortie_stricter, self.oracle_stricter, self.value_divergence);
        if !self.causes.is_empty() {
            let mut s = String::new();
            for (k, v) in &self.causes {
                s.push_str(&format!("{k}={v} "));
            }
            println!("                mortie_stricter by cause: {}", s.trim_end());
        }
        for e in &self.examples {
            println!("                {e}");
        }
    }
}

fn compare(path: &str, names: Option<&str>) {
    let blobs = read_blobs(path);
    let names: Vec<String> = names
        .map(|p| fs::read_to_string(p).unwrap().lines().map(String::from).collect())
        .unwrap_or_default();
    let cap: usize = env::var("DIFF_EXAMPLES").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(6);
    let mut t = [
        ("wkb 0.9", Tally { cap, ..Default::default() }),
        ("geozero Ewkb", Tally { cap, ..Default::default() }),
        ("geozero Wkb", Tally { cap, ..Default::default() }),
    ];
    for (i, b) in blobs.iter().enumerate() {
        let name = names.get(i).map(String::as_str).unwrap_or("-");
        let mine = mortie_rings(b);
        t[0].1.compare(i, name, &mine, &wkb_crate(b));
        t[1].1.compare(i, name, &mine, &geozero_crate(b));
        t[2].1.compare(i, name, &mine, &geozero_plain(b));
    }
    println!("blobs={}", blobs.len());
    for (label, tally) in &t {
        tally.report(label, blobs.len());
    }
}

fn bench(path: &str, repeats: usize) {
    let blobs = read_blobs(path);
    let total: usize = blobs.iter().map(|b| b.len()).sum();
    println!("blobs={} bytes={:.1} MiB repeats={repeats}",
             blobs.len(), total as f64 / (1024.0 * 1024.0));
    let mut rows: Vec<(&str, Vec<f64>)> = Vec::new();
    for (label, f) in [
        ("mortie", &mortie_rings as &dyn Fn(&[u8]) -> Result<Rings, String>),
        ("wkb 0.9", &wkb_crate),
        ("geozero", &geozero_crate),
    ] {
        let mut times = Vec::new();
        for _ in 0..repeats {
            let t0 = std::time::Instant::now();
            let mut acc = 0usize;
            for b in &blobs {
                acc += f(b).map(|r| r.len()).unwrap_or(0);
            }
            std::hint::black_box(acc);
            times.push(t0.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rows.push((label, times));
    }
    let base = rows[0].1[rows[0].1.len() / 2];
    for (label, t) in &rows {
        let med = t[t.len() / 2];
        println!("{label:>10}: min {:7.1} ms  median {:7.1} ms  max {:7.1} ms  \
                  {:6.2} us/blob  {:5.2}x",
                 t[0] * 1e3, med * 1e3, t[t.len() - 1] * 1e3,
                 med * 1e6 / blobs.len() as f64, med / base);
    }
}

fn main() {
    let a: Vec<String> = env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("compare") => compare(&a[2], a.get(3).map(String::as_str)),
        Some("bench") => bench(&a[2], a.get(3).map_or(7, |s| s.parse().unwrap())),
        _ => eprintln!("usage: compare <blobs> [names] | bench <blobs> [repeats]"),
    }
}
'''


def pack(blobs):
    """Serialize blobs as length-prefixed records.

    Parameters
    ----------
    blobs : list of bytes
        The WKB blobs to write.

    Returns
    -------
    bytes
        Each blob preceded by its ``uint32`` little-endian length.
    """
    return b"".join(struct.pack("<I", len(b)) + bytes(b) for b in blobs)


def fixture_blobs():
    """Build the edge-case fixture set, with a name per blob.

    Every dialect the reader claims to handle, in both byte orders and both
    the ISO and EWKB spellings, plus the in-tree Antarctic basins.

    Returns
    -------
    tuple
        ``(blobs, names)``.
    """
    import shapely

    wkts = {
        "polygon": "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))",
        "polygon_hole": ("POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75),"
                         "(20 -74, 30 -74, 30 -72, 20 -72, 20 -74))"),
        "multipolygon": ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)),"
                         "((5 5, 6 5, 6 6, 5 6, 5 5)))"),
        "multipolygon_hole": (
            "MULTIPOLYGON (((10 -75, 40 -75, 40 -71, 10 -71, 10 -75),"
            "(20 -74, 30 -74, 30 -72, 20 -72, 20 -74)),"
            "((0 0, 2 0, 2 2, 0 2, 0 0)))"),
        "antimeridian": "POLYGON ((170 -20, -170 -20, -170 -10, 170 -10, 170 -20))",
        "pole": "POLYGON ((0 -89.5, 90 -89.5, 180 -89.5, -90 -89.5, 0 -89.5))",
        "linestring": "LINESTRING (10 -75, 40 -75, 40 -71)",
        "multilinestring": "MULTILINESTRING ((10 -75, 40 -75), (40 -71, 10 -71))",
        "polygon_z": "POLYGON Z ((10 -75 7, 40 -75 7, 40 -71 7, 10 -71 7, 10 -75 7))",
        "polygon_m": "POLYGON M ((10 -75 1, 40 -75 1, 40 -71 1, 10 -71 1, 10 -75 1))",
        "polygon_zm": ("POLYGON ZM ((10 -75 7 1, 40 -75 7 1, 40 -71 7 1,"
                       " 10 -71 7 1, 10 -75 7 1))"),
        "point": "POINT (10 -75)",
        "multipoint": "MULTIPOINT ((10 -75), (11 -74))",
        "geomcollection": "GEOMETRYCOLLECTION (POINT (10 -75))",
        "polygon_empty": "POLYGON EMPTY",
        "multipolygon_empty": "MULTIPOLYGON EMPTY",
        "linestring_empty": "LINESTRING EMPTY",
    }
    blobs, names = [], []
    for key, wkt in wkts.items():
        g = shapely.from_wkt(wkt)
        for order in (1, 0):
            for flavor in ("iso", "extended"):
                blobs.append(shapely.to_wkb(g, byte_order=order, flavor=flavor))
                names.append(f"{key}/bo{order}/{flavor}")
        blobs.append(shapely.to_wkb(shapely.set_srid(g, 4326), include_srid=True))
        names.append(f"{key}/ewkb_srid")

    # A LinearRing has no WKB type of its own — GEOS writes it as a LineString,
    # which is how the backend-decoded path saw it too.
    blobs.append(shapely.to_wkb(shapely.LinearRing(
        [(10, -75), (40, -75), (40, -71), (10, -75)])))
    names.append("linearring/bo1")

    pts = [(10, -75), (40, -75), (40, -71), (10, -71), (10, -75)]
    part = (struct.pack("<BII", 1, 3 | 0x20000000, 4326)
            + struct.pack("<II", 1, len(pts))
            + b"".join(struct.pack("<dd", x, y) for x, y in pts))
    blobs.append(struct.pack("<BII", 1, 6 | 0x20000000, 4326)
                 + struct.pack("<I", 1) + part)
    names.append("multipolygon/nested_ewkb_srid")
    blobs.append(shapely.to_wkb(shapely.from_wkt(
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), EMPTY)")))
    names.append("multipolygon/empty_part")
    solid = shapely.to_wkb(shapely.from_wkt(wkts["polygon"]))
    blobs.append(solid + b"\xde\xad")
    names.append("polygon/trailing_bytes")

    if BASINS.exists():
        table = np.loadtxt(BASINS)
        for bid in np.unique(table[:, 2]).astype(int):
            m = table[:, 2] == bid
            la, lo = table[m, 0], table[m, 1]
            if la[0] != la[-1] or lo[0] != lo[-1]:
                la, lo = np.append(la, la[0]), np.append(lo, lo[0])
            xy = np.empty(la.size * 2)
            xy[0::2], xy[1::2] = lo, la
            blobs.append(struct.pack("<BIII", 1, 3, 1, la.size)
                         + xy.astype("<f8").tobytes())
            names.append(f"basin/{bid}")
    return blobs, names


def malformed_blobs(n_mutations=40_000, n_noise=5_000):
    """Build the malformed / truncated / fuzzed set, with a name per blob.

    Parameters
    ----------
    n_mutations : int, optional
        Random byte mutations of valid blobs.
    n_noise : int, optional
        Blobs of pure random bytes.

    Returns
    -------
    tuple
        ``(blobs, names)``.
    """
    import shapely

    rng = np.random.default_rng(157)
    solid = shapely.to_wkb(shapely.from_wkt(
        "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75))"))
    multi = shapely.to_wkb(shapely.from_wkt(
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)),((5 5, 6 5, 6 6, 5 6, 5 5)))"))
    holed = shapely.to_wkb(shapely.from_wkt(
        "POLYGON ((10 -75, 40 -75, 40 -71, 10 -71, 10 -75),"
        "(20 -74, 30 -74, 30 -72, 20 -72, 20 -74))"))
    pts = [(10, -75), (40, -75), (40, -71), (10, -71), (10, -75)]

    blobs, names = [], []
    for src, tag in ((solid, "polygon"), (multi, "multipolygon")):
        for cut in range(len(src)):
            blobs.append(src[:cut])
            names.append(f"truncate/{tag}/{cut}")
    for flag in (2, 7, 255):
        b = bytearray(solid)
        b[0] = flag
        blobs.append(bytes(b))
        names.append(f"badflag/{flag}")
    blobs.append(struct.pack("<BIII", 1, 3, 1, 1_000_000_000) + b"\x00" * 32)
    names.append("lying/vertex_count")
    blobs.append(struct.pack("<BII", 1, 3, 1_000_000_000))
    names.append("lying/ring_count")
    blobs.append(struct.pack("<BII", 1, 6, 1_000_000_000))
    names.append("lying/part_count")
    blobs.append(struct.pack("<BIII", 1, 3, 1, 4)
                 + b"".join(struct.pack("<dd", x, y) for x, y in pts[:4]))
    names.append("unclosed/ring")
    nan_ring = list(pts)
    nan_ring[1] = (nan_ring[1][0], float("nan"))
    blobs.append(struct.pack("<BIII", 1, 3, 1, 5)
                 + b"".join(struct.pack("<dd", x, y) for x, y in nan_ring))
    names.append("nan/midring")
    for code in (0, 8, 99, 1_000_000):
        blobs.append(struct.pack("<BII", 1, code, 0))
        names.append(f"badtype/{code}")

    seeds = [solid, multi, holed]
    for i in range(n_mutations):
        src = bytearray(seeds[i % len(seeds)])
        for _ in range(int(rng.integers(1, 4))):
            src[int(rng.integers(0, len(src)))] = int(rng.integers(0, 256))
        blobs.append(bytes(src))
        names.append(f"mutate/{i}")
    for i in range(n_noise):
        size = int(rng.integers(0, 64))
        blobs.append(bytes(rng.integers(0, 256, size, dtype=np.uint8)))
        names.append(f"noise/{i}")
    return blobs, names


def build_harness(work):
    """Generate and compile the throwaway comparison crate.

    Parameters
    ----------
    work : pathlib.Path
        Directory to generate into — deliberately outside the repository, so
        the comparison cannot touch mortie's dependency graph.

    Returns
    -------
    pathlib.Path
        The compiled binary.
    """
    (work / "src").mkdir(parents=True, exist_ok=True)
    (work / "Cargo.toml").write_text(CARGO_TOML)
    (work / "src/main.rs").write_text(MAIN_RS)
    # The real parser, verbatim, minus the submodule that pulls in rayon.
    source = (REPO / "src_rust/src/wkb.rs").read_text()
    stripped = "\n".join(
        line for line in source.splitlines() if line.strip() != "pub mod batch;"
    )
    assert "pub fn parse" in stripped, "wkb.rs did not contain the parser"
    (work / "src/mortie_wkb.rs").write_text(stripped + "\n")
    subprocess.run(["cargo", "build", "--release"], cwd=work, check=True)
    return work / "target/release/mortie-wkb-diff"


def main():
    """Generate the harness, run the comparisons, and print the results.

    Returns
    -------
    int
        ``0`` — the process exit status; disagreements are reported rather
        than turned into a failing exit, since strictness divergences are
        expected findings, not errors.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", help="parquet file with a WKB `geometry` column")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--keep", help="generate into this directory and keep it")
    args = ap.parse_args()

    tmp = None
    if args.keep:
        work = Path(args.keep)
    else:
        tmp = tempfile.mkdtemp(prefix="mortie-wkb-diff-")
        work = Path(tmp)
    try:
        binary = build_harness(work)
        sets = [("fixtures", *fixture_blobs()), ("malformed", *malformed_blobs())]
        if args.corpus:
            import pyarrow.parquet as pq
            col = pq.read_table(args.corpus, columns=["geometry"])["geometry"]
            sets.append(("corpus", col.to_pylist(), None))

        for name, blobs, names in sets:
            bfile = work / f"{name}.blobs"
            bfile.write_bytes(pack(blobs))
            cmd = [str(binary), "compare", str(bfile)]
            if names:
                nfile = work / f"{name}.names"
                nfile.write_text("\n".join(names) + "\n")
                cmd.append(str(nfile))
            print(f"\n===== {name} =====", flush=True)
            subprocess.run(cmd, check=True)
            if name != "malformed":
                subprocess.run([str(binary), "bench", str(bfile),
                                str(args.repeats)], check=True)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
