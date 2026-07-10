//! Cause-tagged `node_straddles` instrumentation (issue #90).
//!
//! Compiled only under the `descent-stats` cargo feature — a cfg flag, not a
//! dependency — so release builds carry none of it and the descent hot path is
//! byte-for-byte unchanged.  The collector answers *why* each straddle verdict
//! fired (the issue #90 cause taxonomy) and records every straddle-**stopped**
//! leaf with enough geometry for an independent Python-side over-refinement
//! test.
//!
//! Protocol (single descent at a time): [`take`] to clear, run one descent,
//! [`take`] again to read.  Concurrent descents from multiple Python threads
//! would interleave their records — the measurement harness runs one at a time.

use std::cell::Cell;
use std::sync::Mutex;

use crate::geo2mort::boundaries_step_scalar;
use crate::morton::nested2mort;
use crate::sphere::{dot, Vec3};

/// Why `node_straddles` returned `true`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// A polygon vertex's leaf cell falls in the cell (clause 1).
    VertexLeaf = 0,
    /// A relevant edge crosses a cell edge of the 4-corner quad (clause 2).
    QuadCross = 1,
    /// The closed-set exact-incidence branch of the quad test
    /// ([`super::edge_touches_cell_edge_degenerate`]'s bit-exact-zero returns).
    /// Deliberate refinement under the #103 contract — **excluded** from the
    /// over-refinement count.
    QuadTouch = 2,
    /// The centre→corner crossing-parity clause (clipped corner/edge).
    CornerParity = 3,
    /// The densified near-pole true-boundary path (issue #32).
    NearPoleBulge = 4,
}

pub const N_CAUSES: usize = 5;

/// One straddle-stopped leaf (refined to the stop rule and emitted as a
/// boundary cell), with everything the measurement needs to re-test the cell
/// independently: the cell as a morton word + depth, the cause, the centre
/// fill state, the centre vector, and the circumradius of the **densified**
/// (true, curved) boundary in radians.
pub struct LeafRecord {
    pub morton: u64,
    pub depth: u8,
    pub cause: Cause,
    pub fill: bool,
    pub center: Vec3,
    pub circ: f64,
}

/// Everything recorded since the last [`take`].
pub struct Stats {
    /// Straddle-stopped leaves per cause.
    pub leaf: [u64; N_CAUSES],
    /// Straddle nodes refined further per cause.
    pub internal: [u64; N_CAUSES],
    pub leaves: Vec<LeafRecord>,
}

impl Stats {
    const fn new() -> Self {
        Stats {
            leaf: [0; N_CAUSES],
            internal: [0; N_CAUSES],
            leaves: Vec::new(),
        }
    }
}

impl Default for Stats {
    fn default() -> Self {
        Stats::new()
    }
}

/// Global collector.  A stdlib `Mutex` (no new dependency) rather than
/// per-worker counters: the descent's rayon workers contend only at straddle
/// events, `record_leaf` updates count and record under one lock (so
/// `sum(leaf) == leaves.len()` holds under any interleaving), and this
/// feature never ships in a release build.
static STATS: Mutex<Stats> = Mutex::new(Stats::new());

thread_local! {
    /// Cause of this thread's most recent `node_straddles == true` verdict.
    /// Same-thread hand-off only: `node_straddles` sets it on its `true`
    /// returns and the caller consumes it immediately via [`take_cause`].
    static LAST_CAUSE: Cell<Option<Cause>> = const { Cell::new(None) };
    /// Whether this thread's most recent `edge_hits_cell_edge == true`
    /// verdict came from the closed-set exact-incidence branch (a touch)
    /// rather than a crossing.  Only written on `true` verdicts, so the
    /// short-circuiting quad loop leaves it set by the deciding hit.
    static LAST_HIT_TOUCH: Cell<bool> = const { Cell::new(false) };
}

/// Tag the most recent `edge_hits_cell_edge == true` verdict: `true` for the
/// closed-set exact-incidence branch, `false` for a crossing.
pub(super) fn note_touch(touch: bool) {
    LAST_HIT_TOUCH.with(|c| c.set(touch));
}

/// Cause for a quad-loop hit, from the deciding `edge_hits_cell_edge` verdict.
pub(super) fn quad_cause() -> Cause {
    if LAST_HIT_TOUCH.with(|c| c.get()) {
        Cause::QuadTouch
    } else {
        Cause::QuadCross
    }
}

/// Record the cause of a `node_straddles == true` verdict (called on each of
/// its `true` return paths).
pub(super) fn set_cause(cause: Cause) {
    LAST_CAUSE.with(|c| c.set(Some(cause)));
}

/// Consume the cause set by the `node_straddles` call that just returned
/// `true`.  Panics if a `true` path failed to tag — a taxonomy gap.
pub(super) fn take_cause() -> Cause {
    LAST_CAUSE
        .with(|c| c.take())
        .expect("node_straddles returned true without tagging a cause")
}

/// Count a straddle node that will be refined further.
pub(super) fn record_internal(cause: Cause) {
    STATS.lock().unwrap().internal[cause as usize] += 1;
}

/// Record a straddle-stopped leaf.  The circumradius is measured against the
/// densified true boundary (step 8 ⇒ 32 points), not the 4-corner quad, so a
/// near-pole cell's bulge is inside it.
pub(super) fn record_leaf(node: &super::Node, cause: Cause) {
    let bnd = boundaries_step_scalar(node.depth, node.pixel, 8);
    let circ = bnd
        .iter()
        .map(|b| dot(&node.center, b).clamp(-1.0, 1.0).acos())
        .fold(0.0_f64, f64::max);
    let mut s = STATS.lock().unwrap();
    s.leaf[cause as usize] += 1;
    s.leaves.push(LeafRecord {
        morton: nested2mort(node.pixel, node.depth),
        depth: node.depth,
        cause,
        fill: node.fill,
        center: node.center,
        circ,
    });
}

/// Return everything recorded since the last call and reset the collector.
pub fn take() -> Stats {
    std::mem::take(&mut *STATS.lock().unwrap())
}
