"""Authalic latitude convention tests (issue #186).

Covers the standalone converters against the high-precision reference table
(``data/authalic_reference.json``), the coherence of the ``latitude=``
parameter across ingress and egress surfaces, the documented divergence
pattern (identical at the equator and poles, maximal ~0.1283 deg at 45 deg),
and the ``"geodetic-spherical"`` legacy escape against pre-change goldens
(``data/legacy_geodetic_goldens.json``, captured from mortie 0.9.7).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mortie
from mortie import _rustie, convert

DATA = Path(__file__).parent / "data"

#: The documented series bound (1e-13 rad) in degrees.
BOUND_DEG = np.degrees(1e-13)

# The legacy-golden input sets (the fixture pins their outputs).
GOLDEN_PTS = [(0.0, 0.0), (45.0, -122.0), (-45.0, 10.0), (30.0, 150.0),
              (-80.0, 120.0), (89.9999, 0.0), (-89.9999, 45.0),
              (22.5, -60.0), (-35.264389, 170.0), (60.0, -45.0)]
GOLDEN_LATS = np.array([p[0] for p in GOLDEN_PTS])
GOLDEN_LONS = np.array([p[1] for p in GOLDEN_PTS])
TRI_LATS = [40.0, 50.0, 45.0]
TRI_LONS = [-120.0, -120.0, -110.0]


@pytest.fixture(scope="module")
def reference():
    return json.loads((DATA / "authalic_reference.json").read_text())


@pytest.fixture(scope="module")
def legacy():
    return json.loads((DATA / "legacy_geodetic_goldens.json").read_text())


class TestConverters:
    """geodetic_to_authalic / authalic_to_geodetic against the references."""

    def test_forward_matches_reference(self, reference):
        lats = np.array([float(row[0]) for row in reference["forward_deg"]])
        want = np.array([float(row[1]) for row in reference["forward_deg"]])
        got = mortie.geodetic_to_authalic(lats)
        assert_allclose(got, want, rtol=0.0, atol=BOUND_DEG)

    def test_inverse_matches_reference(self, reference):
        lats = np.array([float(row[0]) for row in reference["inverse_deg"]])
        want = np.array([float(row[1]) for row in reference["inverse_deg"]])
        got = mortie.authalic_to_geodetic(lats)
        assert_allclose(got, want, rtol=0.0, atol=BOUND_DEG)

    def test_round_trip(self):
        lats = np.linspace(-90.0, 90.0, 3601)
        assert_allclose(
            mortie.authalic_to_geodetic(mortie.geodetic_to_authalic(lats)),
            lats, rtol=0.0, atol=2 * BOUND_DEG,
        )

    def test_fixed_points_exact(self):
        for lat in (-90.0, 0.0, 90.0):
            assert mortie.geodetic_to_authalic(lat) == lat
            assert mortie.authalic_to_geodetic(lat) == lat

    def test_divergence_peaks_at_45(self):
        # ~0.1283 degrees (~14.3 km) equatorward at the 45-degree band.
        d = 45.0 - mortie.geodetic_to_authalic(45.0)
        assert abs(d - 0.12829712656606) < 1e-9

    def test_scalar_and_shape_contract(self):
        assert np.isscalar(mortie.geodetic_to_authalic(45.0))
        one = mortie.geodetic_to_authalic(np.array([45.0]))
        assert one.shape == (1,)
        two_d = mortie.authalic_to_geodetic(np.zeros((3, 4)))
        assert two_d.shape == (3, 4)

    def test_non_finite_propagates(self):
        out = mortie.geodetic_to_authalic(np.array([np.nan, np.inf, 1.0]))
        assert np.isnan(out[0]) and np.isinf(out[1]) and np.isfinite(out[2])


class TestParameterValidation:
    """Invalid latitude= values raise ValueError on every surface."""

    BAD = "geodetic"  # a plausible misspelling

    @pytest.mark.parametrize("call", [
        lambda: mortie.geo2mort(45.0, 0.0, latitude="geodetic"),
        lambda: convert.geo2uniq(45.0, 0.0, latitude="geodetic"),
        lambda: mortie.mort2geo(np.uint64(2**62), latitude="geodetic"),
        lambda: mortie.mort2bbox(np.uint64(2**62), latitude="geodetic"),
        lambda: mortie.mort2polygon(np.uint64(2**62), latitude="geodetic"),
        lambda: mortie.morton_coverage(
            TRI_LATS, TRI_LONS, order=5, latitude="geodetic"),
        lambda: mortie.morton_coverage_moc(
            TRI_LATS, TRI_LONS, order=5, latitude="geodetic"),
        lambda: mortie.ring_is_simple(TRI_LATS, TRI_LONS, latitude="geodetic"),
        lambda: mortie.ring_validity(TRI_LATS, TRI_LONS, latitude="geodetic"),
        lambda: mortie.linestring_coverage(
            TRI_LATS, TRI_LONS, order=5, latitude="geodetic"),
        lambda: mortie.polygons_to_morton_mocs(
            TRI_LATS, TRI_LONS, [0, 3], order=5, latitude="geodetic"),
        lambda: _rustie.rust_dissolve(
            np.array([2**62], dtype=np.uint64), 1, "geodetic"),
    ])
    def test_invalid_value_raises(self, call):
        with pytest.raises(ValueError, match="latitude"):
            call()

    def test_latitude_is_keyword_only(self):
        with pytest.raises(TypeError):
            mortie.mort2geo(np.uint64(2**62), "authalic")


class TestIngressIdentity:
    """latitude="authalic" == converting the latitudes + the legacy escape."""

    def test_geo2mort_grid(self):
        lats = np.linspace(-89.0, 89.0, 91)
        lons = np.linspace(-180.0, 179.0, 91)
        for order in (6, 18, 29):
            direct = mortie.geo2mort(lats, lons, order=order)
            composed = mortie.geo2mort(
                mortie.geodetic_to_authalic(lats), lons, order=order,
                latitude="geodetic-spherical",
            )
            assert_array_equal(direct, composed)

    def test_geo2mort_points_identity(self):
        direct = mortie.geo2mort(GOLDEN_LATS, GOLDEN_LONS)
        composed = mortie.geo2mort(
            mortie.geodetic_to_authalic(GOLDEN_LATS), GOLDEN_LONS,
            latitude="geodetic-spherical",
        )
        assert_array_equal(direct, composed)

    def test_coverage_identity(self):
        direct = mortie.morton_coverage(
            TRI_LATS, TRI_LONS, order=8)
        composed = mortie.morton_coverage(
            mortie.geodetic_to_authalic(np.asarray(TRI_LATS)), TRI_LONS,
            order=8, latitude="geodetic-spherical",
        )
        assert_array_equal(direct, composed)

    def test_conventions_agree_at_equator_and_poles(self):
        lats = np.array([0.0, 90.0, -90.0])
        lons = np.array([12.0, 34.0, 56.0])
        assert_array_equal(
            mortie.geo2mort(lats, lons, order=29),
            mortie.geo2mort(lats, lons, order=29,
                            latitude="geodetic-spherical"),
        )

    def test_conventions_diverge_at_45(self):
        # ~0.1283 deg of latitude is many cells at order 18.
        a = mortie.geo2mort(45.0, 10.0, order=18)
        g = mortie.geo2mort(45.0, 10.0, order=18,
                            latitude="geodetic-spherical")
        assert a[0] != g[0]


class TestEgressCoherence:
    """Egress inverts ingress: decode returns geodetic coordinates."""

    def test_mort2geo_matches_converted_legacy(self):
        words = mortie.geo2mort(GOLDEN_LATS, GOLDEN_LONS, order=18)
        lat_a, lon_a = mortie.mort2geo(words)
        lat_g, lon_g = mortie.mort2geo(words, latitude="geodetic-spherical")
        assert_array_equal(lon_a, lon_g)  # longitude never converts
        assert_allclose(lat_a, mortie.authalic_to_geodetic(lat_g),
                        rtol=0.0, atol=2 * BOUND_DEG)

    def test_geo2mort_mort2geo_round_trip(self):
        # Under both conventions the decoded centre lies within the order-18
        # cell scale (~0.003 deg) of the input point.
        for latitude in ("authalic", "geodetic-spherical"):
            words = mortie.geo2mort(GOLDEN_LATS, GOLDEN_LONS, order=18,
                                    latitude=latitude)
            lat, lon = mortie.mort2geo(words, latitude=latitude)
            assert_allclose(lat, GOLDEN_LATS, rtol=0.0, atol=0.01)
            dlon = (np.asarray(lon) - GOLDEN_LONS + 180.0) % 360.0 - 180.0
            # Longitude cell width grows near the poles; exclude the two
            # near-pole golden points from the tight check.
            body = np.abs(GOLDEN_LATS) < 85.0
            assert np.all(np.abs(dlon[body]) < 0.01)

    def test_mort2polygon_latitudes_convert(self):
        w = int(mortie.geo2mort(45.0, 10.0, order=10)[0])
        ring_a = np.asarray(mortie.mort2polygon(w))
        ring_g = np.asarray(mortie.mort2polygon(
            w, latitude="geodetic-spherical"))
        assert_array_equal(ring_a[:, 1], ring_g[:, 1])  # lon column
        assert_allclose(
            ring_a[:, 0], mortie.authalic_to_geodetic(ring_g[:, 0]),
            rtol=0.0, atol=2 * BOUND_DEG,
        )

    def test_mort2bbox_latitudes_convert(self):
        w = int(mortie.geo2mort(45.0, 10.0, order=10)[0])
        bb_a = mortie.mort2bbox(w)
        bb_g = mortie.mort2bbox(w, latitude="geodetic-spherical")
        assert bb_a["west"] == bb_g["west"] and bb_a["east"] == bb_g["east"]
        assert_allclose(
            [bb_a["south"], bb_a["north"]],
            mortie.authalic_to_geodetic(
                np.array([bb_g["south"], bb_g["north"]])),
            rtol=0.0, atol=2 * BOUND_DEG,
        )

    def test_dissolve_latitudes_convert(self):
        cover = mortie.morton_coverage(TRI_LATS, TRI_LONS, order=6)
        arr = np.ascontiguousarray(cover)
        shells_a, _ = _rustie.rust_dissolve(arr, 1)
        shells_g, _ = _rustie.rust_dissolve(arr, 1, "geodetic-spherical")
        assert len(shells_a) == len(shells_g)
        for ra, rg in zip(shells_a, shells_g):
            ra, rg = np.asarray(ra), np.asarray(rg)
            assert_array_equal(ra[:, 0], rg[:, 0])  # lon column
            assert_allclose(ra[:, 1], mortie.authalic_to_geodetic(rg[:, 1]),
                            rtol=0.0, atol=2 * BOUND_DEG)


class TestLegacyGoldens:
    """latitude="geodetic-spherical" reproduces the pre-change outputs."""

    def test_points_o29(self, legacy):
        got = mortie.geo2mort(GOLDEN_LATS, GOLDEN_LONS,
                              latitude="geodetic-spherical")
        assert_array_equal(got, np.array(legacy["points_o29"],
                                         dtype=np.uint64))

    @pytest.mark.parametrize("order", [6, 12, 18])
    def test_area_words(self, legacy, order):
        got = mortie.geo2mort(GOLDEN_LATS, GOLDEN_LONS, order=order,
                              latitude="geodetic-spherical")
        assert_array_equal(got, np.array(legacy[f"area_o{order}"],
                                         dtype=np.uint64))

    def test_flat_coverage(self, legacy):
        got = mortie.morton_coverage(TRI_LATS, TRI_LONS, order=6,
                                     latitude="geodetic-spherical")
        assert_array_equal(got, np.array(legacy["tri_cov_o6"],
                                         dtype=np.uint64))

    def test_moc_coverage(self, legacy):
        got = mortie.morton_coverage_moc(TRI_LATS, TRI_LONS, order=10,
                                         latitude="geodetic-spherical")
        assert_array_equal(got, np.array(legacy["tri_moc_o10"],
                                         dtype=np.uint64))

    def test_linestring_coverage(self, legacy):
        got = mortie.linestring_coverage(
            [40.0, 50.0, 45.0], [-120.0, -110.0, -100.0], order=8,
            latitude="geodetic-spherical",
        )
        assert_array_equal(got, np.array(legacy["line_o8"], dtype=np.uint64))

    def test_mort2geo(self, legacy):
        words = np.array(legacy["area_o12"], dtype=np.uint64)
        lat, lon = mortie.mort2geo(words, latitude="geodetic-spherical")
        assert_allclose(lat, legacy["mort2geo_o12_lat"], rtol=0.0, atol=0.0)
        assert_allclose(lon, legacy["mort2geo_o12_lon"], rtol=0.0, atol=0.0)


class TestBatchAndGeometryParity:
    """The batch / WKB / geometry skins thread latitude= coherently."""

    @pytest.mark.parametrize("latitude", ["authalic", "geodetic-spherical"])
    def test_polygons_batch_matches_scalar(self, latitude):
        lats = np.array(TRI_LATS + [10.0, 20.0, 15.0])
        lons = np.array(TRI_LONS + [-80.0, -80.0, -70.0])
        values, off = mortie.polygons_to_morton_mocs(
            lats, lons, [0, 3, 6], order=8, latitude=latitude)
        first = values[off[0]:off[1]]
        scalar = mortie.morton_coverage_moc(TRI_LATS, TRI_LONS, order=8,
                                            latitude=latitude)
        assert_array_equal(first, scalar)

    @pytest.mark.parametrize("latitude", ["authalic", "geodetic-spherical"])
    def test_from_wkbs_matches_from_wkb(self, latitude):
        shapely = pytest.importorskip("shapely")
        poly = shapely.Polygon(zip(TRI_LONS, TRI_LATS))
        blob = shapely.to_wkb(poly)
        values, off = mortie.from_wkbs([blob], order=8, latitude=latitude)
        scalar = mortie.from_wkb(blob, order=8, moc=True, latitude=latitude)
        assert_array_equal(values[off[0]:off[1]], scalar)

    def test_wkb_conventions_differ_midlat(self):
        shapely = pytest.importorskip("shapely")
        blob = shapely.to_wkb(shapely.Polygon(zip(TRI_LONS, TRI_LATS)))
        a = mortie.from_wkb(blob, order=8, moc=True)
        g = mortie.from_wkb(blob, order=8, moc=True,
                            latitude="geodetic-spherical")
        assert not np.array_equal(a, g)

    def test_to_geometry_round_trip_coherent(self):
        # Cover -> dissolved outline -> cover again contains the source
        # cells under the default convention end to end.
        pytest.importorskip("shapely")
        cover = mortie.morton_coverage(TRI_LATS, TRI_LONS, order=6)
        outline = mortie.to_geometry(cover, step=4)
        again = mortie.from_geometry(outline, order=6)
        assert set(cover.tolist()) <= set(np.asarray(again).tolist())

    def test_from_latlon_matches_geo2mort(self):
        pytest.importorskip("pandas")
        from mortie.morton_index import MortonIndexArray

        for latitude in ("authalic", "geodetic-spherical"):
            a = MortonIndexArray.from_latlon(
                GOLDEN_LATS, GOLDEN_LONS, order=12, latitude=latitude)
            b = mortie.geo2mort(GOLDEN_LATS, GOLDEN_LONS, order=12,
                                latitude=latitude)
            assert_array_equal(np.asarray(a, dtype=np.uint64), b)
