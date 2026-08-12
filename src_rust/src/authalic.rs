//! Authalic <-> geodetic latitude conversion (issue #186).
//!
//! mortie's spherical HEALPix kernel is equal-area on the *sphere*.  To make
//! cells equal-area on the WGS84 *ellipsoid* by construction, geodetic
//! latitudes are converted to **authalic** latitude before the kernel
//! (ingress) and back after it (egress).  Longitude is unchanged.  The
//! conversion depends only on the pinned reference-ellipsoid constants —
//! WGS84 `a = 6378137`, `1/f = 298.257223563`, fixed across all WGS84
//! realizations — never on the geoid or the epoch.
//!
//! Both directions are 5-harmonic trigonometric series,
//!
//! ```text
//! beta = phi  + FWD_S2*sin(2 phi)  + ... + FWD_S10*sin(10 phi)
//! phi  = beta + INV_S2*sin(2 beta) + ... + INV_S10*sin(10 beta)
//! ```
//!
//! with coefficients that are exact rationals in powers of `e^2` through
//! `e^10`, derived symbolically offline by
//! `mortie/tests/generate_authalic_reference.py` (perturbation expansion of
//! Snyder's closed form `beta = asin(q(phi)/q(pi/2))`, and its series
//! reversion; the e^2..e^6 terms reproduce Snyder 1987 eq. 3-18 exactly).
//! Truncation error measured against the 60-digit closed form on a
//! 0.01-degree grid is `6.2e-15` rad forward and `8.2e-15` rad inverse;
//! with f64 evaluation rounding the documented bound is **`<= 1e-13` rad
//! (~0.6 um on the ground) in each direction**, unit-test enforced against
//! the high-precision references in
//! `mortie/tests/data/authalic_reference.json`.
//!
//! The equator and the poles are exact fixed points of both directions, so
//! the two latitude conventions agree there and diverge in between, maximally
//! at 45 degrees where `|beta - phi| ~= 0.1283` degrees (~14.3 km).

/// WGS84 semi-major axis in meters (exact, by definition).
pub const WGS84_A: f64 = 6378137.0;
/// WGS84 inverse flattening (exact decimal, by definition).
pub const WGS84_INV_F: f64 = 298.257223563;

const F: f64 = 1.0 / WGS84_INV_F;
/// Squared first eccentricity, derived from the pinned constants.
pub const E2: f64 = F * (2.0 - F);
const E4: f64 = E2 * E2;
const E6: f64 = E4 * E2;
const E8: f64 = E4 * E4;
const E10: f64 = E8 * E2;

// Series coefficients: exact rationals in powers of e^2, derived by
// mortie/tests/generate_authalic_reference.py (do not edit by hand).
const FWD_S2: f64 = (-1.0 / 3.0) * E2
    + (-31.0 / 180.0) * E4
    + (-59.0 / 560.0) * E6
    + (-42811.0 / 604800.0) * E8
    + (-605399.0 / 11975040.0) * E10;
const FWD_S4: f64 = (17.0 / 360.0) * E4
    + (61.0 / 1260.0) * E6
    + (76969.0 / 1814400.0) * E8
    + (215431.0 / 5987520.0) * E10;
const FWD_S6: f64 =
    (-383.0 / 45360.0) * E6 + (-3347.0 / 259200.0) * E8 + (-1751791.0 / 119750400.0) * E10;
const FWD_S8: f64 = (6007.0 / 3628800.0) * E8 + (201293.0 / 59875200.0) * E10;
const FWD_S10: f64 = (-5839.0 / 17107200.0) * E10;

const INV_S2: f64 = (1.0 / 3.0) * E2
    + (31.0 / 180.0) * E4
    + (517.0 / 5040.0) * E6
    + (120389.0 / 1814400.0) * E8
    + (1362253.0 / 29937600.0) * E10;
const INV_S4: f64 = (23.0 / 360.0) * E4
    + (251.0 / 3780.0) * E6
    + (102287.0 / 1814400.0) * E8
    + (450739.0 / 9979200.0) * E10;
const INV_S6: f64 =
    (761.0 / 45360.0) * E6 + (47561.0 / 1814400.0) * E8 + (434501.0 / 14968800.0) * E10;
const INV_S8: f64 = (6059.0 / 1209600.0) * E8 + (625511.0 / 59875200.0) * E10;
const INV_S10: f64 = (48017.0 / 29937600.0) * E10;

const FWD: [f64; 5] = [FWD_S2, FWD_S4, FWD_S6, FWD_S8, FWD_S10];
const INV: [f64; 5] = [INV_S2, INV_S4, INV_S6, INV_S8, INV_S10];

/// Evaluate `x + sum_k coeffs[k-1] * sin(2k x)` via the Chebyshev recurrence
/// `sin(2(k+1)x) = 2 cos(2x) sin(2kx) - sin(2(k-1)x)` — one `sin_cos` call
/// total.  Non-finite `x` propagates (NaN in, NaN out).
#[inline]
fn eval_series(x: f64, coeffs: &[f64; 5]) -> f64 {
    let (s2, c2) = (2.0 * x).sin_cos();
    let two_c2 = 2.0 * c2;
    let mut prev = 0.0; // sin(0)
    let mut cur = s2; // sin(2x)
    let mut acc = coeffs[0] * s2;
    for &coeff in &coeffs[1..] {
        let next = two_c2 * cur - prev;
        prev = cur;
        cur = next;
        acc += coeff * next;
    }
    x + acc
}

/// Geodetic -> authalic latitude, radians.
#[inline]
pub fn forward_rad(phi: f64) -> f64 {
    eval_series(phi, &FWD)
}

/// Authalic -> geodetic latitude, radians.
#[inline]
pub fn inverse_rad(beta: f64) -> f64 {
    eval_series(beta, &INV)
}

/// Geodetic -> authalic latitude, degrees.
///
/// `|beta| <= |phi|` holds for the forward direction, so an in-range input
/// stays in range; the clamp only guards the float round-trip at the poles.
/// Non-finite input propagates unchanged (the encode paths map it to their
/// null sentinel downstream, exactly as before).
#[inline]
pub fn forward_deg(lat: f64) -> f64 {
    if !lat.is_finite() {
        return lat;
    }
    forward_rad(lat.to_radians())
        .to_degrees()
        .clamp(-90.0, 90.0)
}

/// Authalic -> geodetic latitude, degrees.
///
/// The inverse pushes latitudes away from the equator (`|phi| >= |beta|`),
/// so a pole-adjacent value could overshoot 90 by a rounding ulp; the clamp
/// restores the contract that in-range input yields in-range output.
#[inline]
pub fn inverse_deg(lat: f64) -> f64 {
    if !lat.is_finite() {
        return lat;
    }
    inverse_rad(lat.to_radians())
        .to_degrees()
        .clamp(-90.0, 90.0)
}

/// The latitude convention of coordinates crossing the mortie API (issue #186).
///
/// `Authalic` (the default everywhere) converts WGS84 geodetic latitudes to
/// authalic latitude at ingress and back at egress, making HEALPix cells
/// equal-area on the ellipsoid by construction.  `GeodeticSpherical` is the
/// pre-0.10 behavior: geodetic latitude fed to the spherical kernel as-is.
/// Cell ids produced under the two conventions are **non-corresponding
/// partitions** — never mix them in one dataset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Latitude {
    /// Equal-area on the WGS84 ellipsoid by construction (default).
    Authalic,
    /// Legacy: geodetic latitude treated as spherical.
    GeodeticSpherical,
}

impl Latitude {
    /// Parse the Python-facing string spelling.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "authalic" => Ok(Self::Authalic),
            "geodetic-spherical" => Ok(Self::GeodeticSpherical),
            other => Err(format!(
                "latitude must be \"authalic\" or \"geodetic-spherical\", got {other:?}"
            )),
        }
    }

    /// Convert one ingress latitude (degrees): geodetic -> kernel frame.
    #[inline]
    pub fn ingress_deg(self, lat: f64) -> f64 {
        match self {
            Self::Authalic => forward_deg(lat),
            Self::GeodeticSpherical => lat,
        }
    }

    /// Convert one egress latitude (degrees): kernel frame -> geodetic.
    #[inline]
    pub fn egress_deg(self, lat: f64) -> f64 {
        match self {
            Self::Authalic => inverse_deg(lat),
            Self::GeodeticSpherical => lat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // High-precision references (mpmath, 60 dps; rounded to nearest f64).
    // Subset of mortie/tests/data/authalic_reference.json — regenerate with
    // mortie/tests/generate_authalic_reference.py.
    const FORWARD_DEG: &[(f64, f64)] = &[
        (-90.0, -90.0),
        (-60.0, -59.888785569885165),
        (-45.0, -44.87170287343394),
        (-30.0, -29.888997034459564),
        (-0.5, -0.497765157038384),
        (0.0, 0.0),
        (1e-7, 9.955300884366168e-08),
        (0.5, 0.497765157038384),
        (5.0, 4.977763096123115),
        (15.0, 14.935956949386629),
        (30.0, 29.888997034459564),
        (35.26438968275, 35.14350666812672),
        (44.9999999, 44.87170277343479),
        (45.0, 44.87170287343394),
        (60.0, 59.888785569885165),
        (75.0, 74.93574548414333),
        (89.0, 88.995513957862),
        (89.9999999, 89.9999998995513),
        (90.0, 90.0),
    ];
    const INVERSE_DEG: &[(f64, f64)] = &[
        (-90.0, -90.0),
        (-60.0, -60.11096559341369),
        (-45.0, -45.12829693352109),
        (-30.0, -30.11125171864826),
        (-0.5, -0.5022448758316342),
        (0.0, 0.0),
        (1e-7, 1.0044899813830867e-07),
        (0.5, 0.5022448758316342),
        (5.0, 5.022335225436584),
        (15.0, 15.064291967518724),
        (30.0, 30.11125171864826),
        (35.26438968275, 35.38545314482135),
        (44.9999999, 45.12829683352225),
        (45.0, 45.12829693352109),
        (60.0, 60.11096559341369),
        (75.0, 75.06400584025931),
        (89.0, 89.00446601553388),
        (89.9999999, 89.9999999004467),
        (90.0, 90.0),
    ];

    /// The documented series bound, in degrees (1e-13 rad).
    const BOUND_DEG: f64 = 1e-13 * 180.0 / PI;

    #[test]
    fn forward_matches_reference() {
        for &(lat, want) in FORWARD_DEG {
            let got = forward_deg(lat);
            assert!(
                (got - want).abs() <= BOUND_DEG,
                "forward({lat}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn inverse_matches_reference() {
        for &(lat, want) in INVERSE_DEG {
            let got = inverse_deg(lat);
            assert!(
                (got - want).abs() <= BOUND_DEG,
                "inverse({lat}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn fixed_points_are_exact() {
        for lat in [0.0, 90.0, -90.0] {
            assert_eq!(forward_deg(lat), lat, "forward({lat})");
            assert_eq!(inverse_deg(lat), lat, "inverse({lat})");
        }
    }

    #[test]
    fn round_trip_dense_grid() {
        // forward . inverse and inverse . forward within the combined bound,
        // every 0.01 degree.
        for i in -9000..=9000 {
            let lat = f64::from(i) / 100.0;
            let there_back = inverse_deg(forward_deg(lat));
            let back_there = forward_deg(inverse_deg(lat));
            assert!(
                (there_back - lat).abs() <= 2.0 * BOUND_DEG,
                "inv(fwd({lat})) = {there_back}"
            );
            assert!(
                (back_there - lat).abs() <= 2.0 * BOUND_DEG,
                "fwd(inv({lat})) = {back_there}"
            );
        }
    }

    #[test]
    fn forward_shrinks_inverse_grows() {
        // |beta| <= |phi| (forward), |phi| >= |beta| (inverse), strict away
        // from the fixed points; both stay inside [-90, 90].
        for i in 1..9000 {
            let lat = f64::from(i) / 100.0;
            let f = forward_deg(lat);
            let g = inverse_deg(lat);
            assert!(f < lat && f > 0.0, "forward({lat}) = {f}");
            assert!(g > lat && g <= 90.0, "inverse({lat}) = {g}");
            assert_eq!(forward_deg(-lat), -f, "forward odd symmetry at {lat}");
            assert_eq!(inverse_deg(-lat), -g, "inverse odd symmetry at {lat}");
        }
    }

    #[test]
    fn max_divergence_at_45() {
        // ~0.1283 degrees (~14.3 km) at the 45-degree band.
        let d = 45.0 - forward_deg(45.0);
        assert!((d - 0.12829712656606).abs() < 1e-9, "divergence {d}");
    }

    #[test]
    fn non_finite_propagates() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!forward_deg(bad).is_finite(), "forward({bad})");
            assert!(!inverse_deg(bad).is_finite(), "inverse({bad})");
        }
    }

    #[test]
    fn parse_conventions() {
        assert_eq!(Latitude::parse("authalic"), Ok(Latitude::Authalic));
        assert_eq!(
            Latitude::parse("geodetic-spherical"),
            Ok(Latitude::GeodeticSpherical)
        );
        assert!(Latitude::parse("geodetic").is_err());
        assert!(Latitude::parse("").is_err());
    }

    #[test]
    fn geodetic_spherical_is_identity() {
        for lat in [-90.0, -45.0, 0.0, 33.3, 90.0] {
            assert_eq!(Latitude::GeodeticSpherical.ingress_deg(lat), lat);
            assert_eq!(Latitude::GeodeticSpherical.egress_deg(lat), lat);
        }
    }
}
