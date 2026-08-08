//! Backend-free WKB → ring arrays (issue #157).
//!
//! mortie's WKB ingest used to delegate the decode to a Python geometry
//! backend (shapely, or spherely as a decode-only fallback) and then lean on
//! shapely's ring introspection to reach the coverage kernels — so a
//! "numpy-only runtime" was true only for callers who never touched WKB.  This
//! module parses the bytes directly: WKB is a small, stable, well-specified
//! binary format, and the descent that consumes the rings is already here.
//!
//! # Output contract
//!
//! The rings come out exactly as `mortie.geometry.decompose` documents them:
//! the exterior **and** interior rings of **every** part, flattened into one
//! ragged array in arrow list layout (`ring i` is
//! `lats[offsets[i]..offsets[i+1]]`), so mortie's even-odd descent unions
//! disjoint outers and carves holes in the one pass.  Vertices are kept as
//! authored — closed rings stay closed, winding is untouched.
//!
//! # Coordinate order
//!
//! WKB stores `(x, y) = (lon, lat)` degrees; mortie's kernels take
//! `(lats, lons)`.  The swap happens once, in [`read_points`]: `x → lons`,
//! `y → lats`.  It is pinned by an asymmetric fixture in the tests (a ring
//! whose latitude and longitude ranges cannot be confused).
//!
//! # Dialects
//!
//! Both byte orders, per geometry header (a MultiPolygon may mix them).
//! Dimensionality is read from either spelling — ISO's `+1000`/`+2000`/`+3000`
//! type offsets or EWKB's `0x8000_0000`/`0x4000_0000` flag bits — and Z/M
//! ordinates are read past and dropped (mortie is 2-D lon/lat).  An EWKB SRID
//! (flag `0x2000_0000`) is skipped, mirroring `_strip_ewkt_srid` on the WKT
//! side: it is advisory, mortie's contract is always EPSG:4326.
//!
//! Trailing bytes after a complete geometry are ignored, matching GEOS (and
//! therefore the shapely-backed path this replaces).

/// Which coverage family a parsed geometry belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Polygon / MultiPolygon — rings for the even-odd descent.
    Polygonal,
    /// LineString / LinearRing / MultiLineString — lines for linestring coverage.
    Linear,
}

impl Kind {
    /// The name this kind carries across to Python (`decompose`'s first element).
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Polygonal => "polygonal",
            Kind::Linear => "linear",
        }
    }
}

/// A parsed geometry as ragged ring arrays, in `(lats, lons)` degree order.
#[derive(Debug)]
pub struct Rings {
    /// Polygonal (rings) or linear (lines).
    pub kind: Kind,
    /// All rings' vertex latitudes, concatenated.
    pub lats: Vec<f64>,
    /// All rings' vertex longitudes, concatenated.
    pub lons: Vec<f64>,
    /// Arrow list offsets: ring `i` is `lats[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<i64>,
}

const WKB_POINT: u32 = 1;
const WKB_LINESTRING: u32 = 2;
const WKB_POLYGON: u32 = 3;
const WKB_MULTIPOINT: u32 = 4;
const WKB_MULTILINESTRING: u32 = 5;
const WKB_MULTIPOLYGON: u32 = 6;
const WKB_GEOMETRYCOLLECTION: u32 = 7;

/// EWKB flag bits: Z ordinate, M ordinate, embedded SRID.
const EWKB_Z: u32 = 0x8000_0000;
const EWKB_M: u32 = 0x4000_0000;
const EWKB_SRID: u32 = 0x2000_0000;

/// A bounds-checked reader over one WKB blob.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Take `n` bytes, or report the truncation with the offset that ran out.
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.remaining() < n {
            return Err(format!(
                "truncated WKB: {n} more byte(s) needed at offset {}, {} remain",
                self.pos,
                self.remaining()
            ));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self, le: bool) -> Result<u32, String> {
        let b: [u8; 4] = self.take(4)?.try_into().expect("take(4) yields 4 bytes");
        Ok(if le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn f64(&mut self, le: bool) -> Result<f64, String> {
        let b: [u8; 8] = self.take(8)?.try_into().expect("take(8) yields 8 bytes");
        Ok(if le {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        })
    }
}

/// One geometry's header: byte order, base type, and coordinate stride.
struct Header {
    /// `true` for little-endian (NDR) bodies, `false` for big-endian (XDR).
    le: bool,
    /// The 1-7 base geometry type, with all dimension/SRID decoration removed.
    base: u32,
    /// Ordinates per vertex: 2, 3 (Z or M), or 4 (ZM).  Only the first two are
    /// kept.
    stride: usize,
}

/// Read a geometry header, resolving both the ISO and EWKB type spellings.
///
/// An EWKB SRID is consumed here and discarded.  Nested geometries (the parts
/// of a Multi\*) carry their own header, so each part is read through this
/// same function and may differ in byte order or dimensionality.
fn read_header(cur: &mut Cursor) -> Result<Header, String> {
    let flag = cur.u8()?;
    let le = match flag {
        0 => false,
        1 => true,
        other => {
            return Err(format!(
                "invalid WKB byte-order flag {other} at offset {}: expected 0 (big-endian) \
                 or 1 (little-endian)",
                cur.pos - 1
            ))
        }
    };
    let raw = cur.u32(le)?;
    let mut stride = 2usize;
    if raw & EWKB_Z != 0 {
        stride += 1;
    }
    if raw & EWKB_M != 0 {
        stride += 1;
    }
    let plain = raw & !(EWKB_Z | EWKB_M | EWKB_SRID);
    // ISO spelling: 1000 + t = Z, 2000 + t = M, 3000 + t = ZM.
    match plain / 1000 {
        0 => {}
        1 | 2 => stride += 1,
        3 => stride += 2,
        _ => return Err(format!("unrecognized WKB geometry type code {raw}")),
    }
    let base = plain % 1000;
    if raw & EWKB_SRID != 0 {
        cur.u32(le)?; // advisory; mortie's contract is always EPSG:4326
    }
    Ok(Header { le, base, stride })
}

/// The name of a WKB base type, for error messages.
fn type_name(base: u32) -> &'static str {
    match base {
        WKB_POINT => "Point",
        WKB_LINESTRING => "LineString",
        WKB_POLYGON => "Polygon",
        WKB_MULTIPOINT => "MultiPoint",
        WKB_MULTILINESTRING => "MultiLineString",
        WKB_MULTIPOLYGON => "MultiPolygon",
        WKB_GEOMETRYCOLLECTION => "GeometryCollection",
        _ => "unknown",
    }
}

/// Accumulates rings into the ragged `(lats, lons, offsets)` layout.
struct Acc {
    lats: Vec<f64>,
    lons: Vec<f64>,
    offsets: Vec<i64>,
}

impl Acc {
    fn new() -> Self {
        Acc {
            lats: Vec::new(),
            lons: Vec::new(),
            offsets: vec![0],
        }
    }
}

/// Read one point sequence (a ring or a line) and close off its ragged entry.
///
/// This is where the axis swap lives: WKB's `x` is longitude and `y` is
/// latitude, so `x` lands in `lons` and `y` in `lats`.  Z/M ordinates are read
/// past and dropped.
fn read_points(cur: &mut Cursor, h: &Header, acc: &mut Acc) -> Result<(), String> {
    let n = cur.u32(h.le)? as usize;
    // Reject an implausible count before reserving against it: a corrupt
    // length field would otherwise ask for an arbitrary allocation.
    let need = n
        .checked_mul(h.stride * 8)
        .ok_or_else(|| format!("WKB vertex count {n} overflows the addressable range"))?;
    if cur.remaining() < need {
        return Err(format!(
            "truncated WKB: {n} vertices need {need} bytes at offset {}, {} remain",
            cur.pos,
            cur.remaining()
        ));
    }
    acc.lats.reserve(n);
    acc.lons.reserve(n);
    for _ in 0..n {
        let x = cur.f64(h.le)?;
        let y = cur.f64(h.le)?;
        for _ in 2..h.stride {
            cur.f64(h.le)?; // Z / M: mortie is 2-D lon/lat
        }
        acc.lons.push(x);
        acc.lats.push(y);
    }
    acc.offsets.push(acc.lats.len() as i64);
    Ok(())
}

/// Read one polygon body: a ring count, then that many rings (exterior first).
fn read_polygon(cur: &mut Cursor, h: &Header, acc: &mut Acc) -> Result<(), String> {
    let n_rings = cur.u32(h.le)? as usize;
    // Each ring is at least its own 4-byte vertex count.
    if cur.remaining() < n_rings.saturating_mul(4) {
        return Err(format!(
            "truncated WKB: {n_rings} rings declared at offset {} with only {} bytes left",
            cur.pos,
            cur.remaining()
        ));
    }
    for _ in 0..n_rings {
        read_points(cur, h, acc)?;
    }
    Ok(())
}

/// Read the `n` parts of a Multi\*, each with its own header, of `want` type.
fn read_parts(
    cur: &mut Cursor,
    h: &Header,
    acc: &mut Acc,
    want: u32,
    polygonal: bool,
) -> Result<(), String> {
    let n_parts = cur.u32(h.le)? as usize;
    // Each part is at least its own 5-byte header.
    if cur.remaining() < n_parts.saturating_mul(5) {
        return Err(format!(
            "truncated WKB: {n_parts} parts declared at offset {} with only {} bytes left",
            cur.pos,
            cur.remaining()
        ));
    }
    for i in 0..n_parts {
        let ph = read_header(cur)?;
        if ph.base != want {
            return Err(format!(
                "malformed WKB: {} part {i} is a {} (type {}), expected a {}",
                type_name(h.base),
                type_name(ph.base),
                ph.base,
                type_name(want)
            ));
        }
        if polygonal {
            read_polygon(cur, &ph, acc)?;
        } else {
            read_points(cur, &ph, acc)?;
        }
    }
    Ok(())
}

/// Parse WKB (or EWKB) bytes into mortie coverage inputs.
///
/// Returns the rings of a Polygon / MultiPolygon (exterior and interior, every
/// part, flattened) or the lines of a LineString / LinearRing /
/// MultiLineString, with `(x, y)` unswapped into `(lats, lons)` degrees.
///
/// # Errors
/// A truncated or structurally malformed blob, an unsupported geometry type
/// (Point, MultiPoint, GeometryCollection — coverage has no meaning for them),
/// or an empty geometry, each named in the message.
pub fn parse(bytes: &[u8]) -> Result<Rings, String> {
    let mut cur = Cursor::new(bytes);
    let h = read_header(&mut cur)?;
    let mut acc = Acc::new();
    let kind = match h.base {
        WKB_POLYGON => {
            read_polygon(&mut cur, &h, &mut acc)?;
            Kind::Polygonal
        }
        WKB_MULTIPOLYGON => {
            read_parts(&mut cur, &h, &mut acc, WKB_POLYGON, true)?;
            Kind::Polygonal
        }
        // A LinearRing has no WKB type of its own — GEOS writes it as a
        // LineString, which is how the shapely-backed path saw it too.
        WKB_LINESTRING => {
            read_points(&mut cur, &h, &mut acc)?;
            Kind::Linear
        }
        WKB_MULTILINESTRING => {
            read_parts(&mut cur, &h, &mut acc, WKB_LINESTRING, false)?;
            Kind::Linear
        }
        other => {
            return Err(format!(
                "unsupported WKB geometry type {other} ({}) for coverage; expected \
                 Polygon, MultiPolygon, LineString, or MultiLineString",
                type_name(other)
            ))
        }
    };
    if acc.lats.is_empty() {
        return Err("empty geometry has no coverage".to_string());
    }
    Ok(Rings {
        kind,
        lats: acc.lats,
        lons: acc.lons,
        offsets: acc.offsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a little-endian WKB blob from a type code and body writer.
    struct Wkb {
        bytes: Vec<u8>,
        le: bool,
    }

    impl Wkb {
        fn new(le: bool) -> Self {
            Wkb {
                bytes: Vec::new(),
                le,
            }
        }
        fn header(mut self, code: u32) -> Self {
            self.bytes.push(if self.le { 1 } else { 0 });
            self.u32(code)
        }
        fn u32(mut self, v: u32) -> Self {
            if self.le {
                self.bytes.extend_from_slice(&v.to_le_bytes());
            } else {
                self.bytes.extend_from_slice(&v.to_be_bytes());
            }
            self
        }
        fn f64(mut self, v: f64) -> Self {
            if self.le {
                self.bytes.extend_from_slice(&v.to_le_bytes());
            } else {
                self.bytes.extend_from_slice(&v.to_be_bytes());
            }
            self
        }
        /// Append a ring of `(lon, lat)` pairs, with `extra` filler ordinates.
        fn ring(mut self, pts: &[(f64, f64)], extra: usize) -> Self {
            self = self.u32(pts.len() as u32);
            for (lon, lat) in pts {
                self = self.f64(*lon).f64(*lat);
                for k in 0..extra {
                    self = self.f64(1000.0 + k as f64);
                }
            }
            self
        }
    }

    /// A ring whose latitude and longitude ranges cannot be confused: lats in
    /// [-75, -71], lons in [10, 40].  Swapping the axes anywhere in the reader
    /// makes both assertions below fail loudly rather than plausibly.
    fn asymmetric() -> Vec<(f64, f64)> {
        vec![
            (10.0, -75.0),
            (40.0, -75.0),
            (40.0, -71.0),
            (10.0, -71.0),
            (10.0, -75.0),
        ]
    }

    fn polygon_wkb(le: bool, code: u32, extra: usize, rings: &[Vec<(f64, f64)>]) -> Vec<u8> {
        let mut w = Wkb::new(le).header(code).u32(rings.len() as u32);
        for r in rings {
            w = w.ring(r, extra);
        }
        w.bytes
    }

    #[test]
    fn coordinate_order_is_x_lon_y_lat() {
        let pts = asymmetric();
        let out = parse(&polygon_wkb(
            true,
            WKB_POLYGON,
            0,
            std::slice::from_ref(&pts),
        ))
        .unwrap();
        assert_eq!(out.kind, Kind::Polygonal);
        assert_eq!(out.offsets, vec![0, 5]);
        // y → lats (all southern), x → lons (all eastern).  Both directions:
        // every lat is in the lat range and every lon in the lon range, so a
        // swapped read cannot pass either check.
        assert_eq!(out.lats, pts.iter().map(|p| p.1).collect::<Vec<_>>());
        assert_eq!(out.lons, pts.iter().map(|p| p.0).collect::<Vec<_>>());
        assert!(out.lats.iter().all(|v| (-75.0..=-71.0).contains(v)));
        assert!(out.lons.iter().all(|v| (10.0..=40.0).contains(v)));
    }

    #[test]
    fn byte_orders_agree() {
        let pts = asymmetric();
        let little = parse(&polygon_wkb(
            true,
            WKB_POLYGON,
            0,
            std::slice::from_ref(&pts),
        ))
        .unwrap();
        let big = parse(&polygon_wkb(false, WKB_POLYGON, 0, &[pts])).unwrap();
        assert_eq!(little.lats, big.lats);
        assert_eq!(little.lons, big.lons);
        assert_eq!(little.offsets, big.offsets);
    }

    #[test]
    fn z_and_m_ordinates_are_dropped() {
        let pts = asymmetric();
        let plain = parse(&polygon_wkb(
            true,
            WKB_POLYGON,
            0,
            std::slice::from_ref(&pts),
        ))
        .unwrap();
        // ISO spellings: 1003 = Z, 2003 = M, 3003 = ZM.
        for (code, extra) in [(1003u32, 1usize), (2003, 1), (3003, 2)] {
            let out = parse(&polygon_wkb(true, code, extra, std::slice::from_ref(&pts))).unwrap();
            assert_eq!(out.lats, plain.lats, "code {code}");
            assert_eq!(out.lons, plain.lons, "code {code}");
        }
        // EWKB flag spellings: Z, M, and both.
        for (code, extra) in [
            (WKB_POLYGON | EWKB_Z, 1usize),
            (WKB_POLYGON | EWKB_M, 1),
            (WKB_POLYGON | EWKB_Z | EWKB_M, 2),
        ] {
            let out = parse(&polygon_wkb(true, code, extra, std::slice::from_ref(&pts))).unwrap();
            assert_eq!(out.lats, plain.lats, "code {code:#x}");
            assert_eq!(out.lons, plain.lons, "code {code:#x}");
        }
    }

    #[test]
    fn ewkb_srid_is_stripped() {
        let pts = asymmetric();
        let plain = parse(&polygon_wkb(
            true,
            WKB_POLYGON,
            0,
            std::slice::from_ref(&pts),
        ))
        .unwrap();
        let mut w = Wkb::new(true)
            .header(WKB_POLYGON | EWKB_SRID)
            .u32(4326)
            .u32(1);
        w = w.ring(&pts, 0);
        let out = parse(&w.bytes).unwrap();
        assert_eq!(out.lats, plain.lats);
        assert_eq!(out.lons, plain.lons);
    }

    #[test]
    fn interior_rings_and_parts_are_all_flattened() {
        let outer = asymmetric();
        let hole = vec![
            (20.0, -74.0),
            (30.0, -74.0),
            (30.0, -72.0),
            (20.0, -72.0),
            (20.0, -74.0),
        ];
        let out = parse(&polygon_wkb(
            true,
            WKB_POLYGON,
            0,
            &[outer.clone(), hole.clone()],
        ))
        .unwrap();
        assert_eq!(out.offsets, vec![0, 5, 10]);
        assert_eq!(&out.lons[5..10], &[20.0, 30.0, 30.0, 20.0, 20.0]);

        // MultiPolygon: two parts, one with a hole — every ring of every part,
        // in order, is what the even-odd descent expects.
        let part_a = polygon_wkb(true, WKB_POLYGON, 0, &[outer.clone(), hole]);
        let part_b = polygon_wkb(false, WKB_POLYGON, 0, &[outer]); // mixed endianness
        let mut multi = Wkb::new(true).header(WKB_MULTIPOLYGON).u32(2).bytes;
        multi.extend_from_slice(&part_a[..]);
        multi.extend_from_slice(&part_b[..]);
        let out = parse(&multi).unwrap();
        assert_eq!(out.kind, Kind::Polygonal);
        assert_eq!(out.offsets, vec![0, 5, 10, 15]);
        assert!(out.lats.iter().all(|v| (-75.0..=-71.0).contains(v)));
    }

    #[test]
    fn linear_geometries_report_linear_kind() {
        let pts = asymmetric();
        let line = Wkb::new(true).header(WKB_LINESTRING).ring(&pts, 0).bytes;
        let out = parse(&line).unwrap();
        assert_eq!(out.kind, Kind::Linear);
        assert_eq!(out.offsets, vec![0, 5]);

        let mut multi = Wkb::new(true).header(WKB_MULTILINESTRING).u32(2).bytes;
        multi.extend_from_slice(&line[..]);
        multi.extend_from_slice(&line[..]);
        let out = parse(&multi).unwrap();
        assert_eq!(out.kind, Kind::Linear);
        assert_eq!(out.offsets, vec![0, 5, 10]);
    }

    #[test]
    fn trailing_bytes_are_ignored() {
        let pts = asymmetric();
        let mut blob = polygon_wkb(true, WKB_POLYGON, 0, &[pts]);
        let clean = parse(&blob).unwrap();
        blob.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        let out = parse(&blob).unwrap();
        assert_eq!(out.lats, clean.lats);
    }

    #[test]
    fn unsupported_types_are_named() {
        for (code, name) in [
            (WKB_POINT, "Point"),
            (WKB_MULTIPOINT, "MultiPoint"),
            (WKB_GEOMETRYCOLLECTION, "GeometryCollection"),
        ] {
            let blob = Wkb::new(true).header(code).u32(0).bytes;
            let err = parse(&blob).unwrap_err();
            assert!(err.contains(name), "{err}");
            assert!(err.contains("unsupported"), "{err}");
        }
    }

    #[test]
    fn empty_geometries_are_rejected() {
        // POLYGON EMPTY (zero rings) and MULTIPOLYGON EMPTY (zero parts).
        let err = parse(&polygon_wkb(true, WKB_POLYGON, 0, &[])).unwrap_err();
        assert_eq!(err, "empty geometry has no coverage");
        let blob = Wkb::new(true).header(WKB_MULTIPOLYGON).u32(0).bytes;
        assert_eq!(parse(&blob).unwrap_err(), "empty geometry has no coverage");
        // A ring that declares zero vertices is empty too.
        let blob = polygon_wkb(true, WKB_POLYGON, 0, &[vec![]]);
        assert_eq!(parse(&blob).unwrap_err(), "empty geometry has no coverage");
    }

    #[test]
    fn malformed_blobs_are_rejected_cleanly() {
        let pts = asymmetric();
        let blob = polygon_wkb(true, WKB_POLYGON, 0, &[pts]);
        // Empty input, and a bad byte-order flag.
        assert!(parse(&[]).unwrap_err().contains("truncated"));
        let mut bad = blob.clone();
        bad[0] = 7;
        assert!(bad_flag(&bad));
        // Truncation at every prefix length is an error, never a panic.
        for cut in 1..blob.len() {
            let err = parse(&blob[..cut]).unwrap_err();
            assert!(err.contains("truncated"), "cut {cut}: {err}");
        }
        // A ring count that would overrun the buffer.
        let mut lying = Wkb::new(true).header(WKB_POLYGON).u32(1).bytes;
        lying.extend_from_slice(&(1_000_000_000u32).to_le_bytes());
        lying.extend_from_slice(&[0u8; 16]);
        assert!(parse(&lying).unwrap_err().contains("truncated"));
        // A part count that would overrun the buffer.
        let lying = Wkb::new(true)
            .header(WKB_MULTIPOLYGON)
            .u32(1_000_000_000)
            .bytes;
        assert!(parse(&lying).unwrap_err().contains("truncated"));
        // A MultiPolygon whose part is a LineString.
        let mut wrong = Wkb::new(true).header(WKB_MULTIPOLYGON).u32(1).bytes;
        wrong.extend_from_slice(&Wkb::new(true).header(WKB_LINESTRING).ring(&[], 0).bytes);
        let err = parse(&wrong).unwrap_err();
        assert!(err.contains("expected a Polygon"), "{err}");
        // An unrecognized dimension decoration.
        let blob = polygon_wkb(true, 9003, 0, &[asymmetric()]);
        assert!(parse(&blob).unwrap_err().contains("unrecognized"));
    }

    fn bad_flag(bytes: &[u8]) -> bool {
        parse(bytes)
            .unwrap_err()
            .contains("invalid WKB byte-order flag")
    }
}
