//! Largest Lyapunov exponent (two-trajectory method), Poincaré section and the
//! chaotic / regular classifier.
//!
//! The renormalisation inside [`largest_lyapunov`] is *real*: the perturbed
//! trajectory is re-integrated from the rescaled state at every checkpoint
//! (the intended Benettin scheme). Renormalising after the reference
//! integration has finished — editing the stored solution — has no effect on
//! the trajectory, so the naive variant of this method telescopes the log-sum
//! down to a single final-separation measurement.

use std::collections::HashSet;
use std::fmt;

use crate::dynamics::{G, State, double_pendulum};
use crate::integrator::{IntegratorError, integrate};

/// Solver tolerances for the integrations below.
///
/// The two-trajectory Lyapunov estimate measures the separation `d ≈ δ₀` of
/// two nearby trajectories, which is only meaningful when the integrator's
/// own trajectory error is much smaller than `δ₀ = 1e-8`. At looser
/// tolerances (`rtol = atol = 1e-9`) the separation sits at the edge of that
/// regime and the estimate is biased high. With `1e-12` the deviation is
/// resolved with a comfortable margin.
const RTOL: f64 = 1e-12;
const ATOL: f64 = 1e-12;

/// Tuning parameters for [`largest_lyapunov`].
#[derive(Clone, Copy, Debug)]
pub struct LyapunovParams {
    /// Integration horizon in seconds.
    pub t: f64,
    /// Output-grid spacing in seconds.
    pub dt: f64,
    /// Initial separation `δ₀` of the perturbed trajectory.
    pub δ0: f64,
    /// Time between renormalisations in seconds.
    pub renorm: f64,
}

impl Default for LyapunovParams {
    fn default() -> Self {
        Self {
            // The tail-difference estimator (see [`largest_lyapunov`]) has a
            // noise floor that decays roughly like 1/√T, but not
            // monotonically — it is a noisy quantity, not a smooth bound.
            // Measured over a 64-point grid of low-energy regular starts, the
            // worst residual λ is ≈ 0.012 at T = 400 and ≈ 0.004 at T = 800,
            // against the 0.015 chaotic threshold. T = 400 classifies every
            // start sampled correctly, but the margin is only ~1.3×; raise
            // this to 800 if you need headroom, at 2× the runtime. Genuinely
            // chaotic starts sit at λ ≈ 1.1 and are never close to the
            // threshold.
            t: 400.0,
            dt: 0.02,
            δ0: 1e-8,
            renorm: 2.0,
        }
    }
}

/// Tuning parameters for [`poincare_section`].
#[derive(Clone, Copy, Debug)]
pub struct PoincareParams {
    /// Integration horizon in seconds.
    pub t: f64,
    /// Output-grid spacing in seconds.
    pub dt: f64,
}

impl Default for PoincareParams {
    fn default() -> Self {
        Self { t: 200.0, dt: 0.01 }
    }
}

/// Outcome of [`classify`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Classification {
    Chaotic,
    Periodic,
    Quasiperiodic,
    /// Regular, but too few Poincaré crossings were collected in the horizon.
    NeedsLongerIntegration,
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chaotic => write!(f, "chaotic"),
            Self::Periodic => write!(f, "periodic"),
            Self::Quasiperiodic => write!(f, "quasiperiodic"),
            Self::NeedsLongerIntegration => write!(f, "regular (need longer integration)"),
        }
    }
}

/// Result of [`classify`]; `points` carries the Poincaré section for regular
/// orbits and is `None` for chaotic ones.
#[derive(Debug)]
pub struct ClassificationResult {
    pub classification: Classification,
    pub λ: f64,
    pub points: Option<Vec<[f64; 2]>>,
}

/// Largest Lyapunov exponent `λ₁` via the two-trajectory (Benettin) method.
///
/// A reference trajectory is integrated once over `[0, t]`; a second
/// trajectory starts `δ₀` away in `θ₁` and is re-integrated from a rescaled
/// state at every `renorm` interval, accumulating `log(d/δ₀)` — this keeps
/// the separation small so that `λ₁ ≈ (1/t) Σ log(d/δ₀)` even when the
/// trajectories would otherwise decorrelate completely.
///
/// For a regular orbit the log-sum does not grow systematically — the
/// tangent-space stretching averages out, but the sum still wanders, driven
/// by step-sequence artefacts that do not average out at these horizons
/// (measured over the demo start's checkpoints: ≈ 3.6 at t = 50, 5.2 at
/// t = 100, 5.5 at t = 150, 6.5 at t = 350, 6.4 at t = 400 — a noisy walk,
/// not a bounded constant). The plain average `log_sum / t` therefore decays
/// like a noisy `const(t)/t` and cannot be told apart from a genuinely small
/// growth rate: at a 100 s horizon the residual sits at `~1.5/100 = 0.015`,
/// exactly the chaotic threshold, so every regular orbit read as chaotic.
/// The estimate therefore measures the **growth** of the log-sum: only
/// checkpoints in the second half of the run contribute (`λ = tail_sum /
/// elapsed`), so a wandering-but-sublinear log-sum yields ≈ 0 regardless of
/// its level while a chaotic one keeps its positive growth rate.
///
/// # Errors
///
/// Returns the integrator error if any of the integrations fail.
pub fn largest_lyapunov(y0: State, p: LyapunovParams) -> Result<f64, IntegratorError> {
    let f = |t: f64, y: &[f64; 4]| double_pendulum(t, State::from_array(*y), G).to_array();
    largest_lyapunov_with(y0, p, |y, t_eval| integrate(f, y, t_eval, RTOL, ATOL))
}

/// Same as [`largest_lyapunov`], but with the integrations delegated to the
/// supplied closure so callers can swap in their own ODE engine. The
/// benchmark uses this to run the *identical* algorithm on every backend
/// instead of maintaining a drift-prone copy.
///
/// The closure receives a state and a time grid and must return **one row per
/// grid entry, in order** — the algorithm indexes `ref_sol[i]` and `seg[1]`
/// directly. A closure that returns fewer rows than grid entries will panic
/// (index out of bounds) rather than report an error.
/// The closure receives a state and a time grid and must return **one row per
/// grid entry, in order** — the algorithm indexes `ref_sol[i]` and `seg[1]`
/// directly. A closure that returns fewer rows than grid entries will panic
/// (index out of bounds) rather than report an error.
///
/// # Errors
///
/// Returns the error from the supplied `integrate_fn` if any integration
/// fails.
pub fn largest_lyapunov_with<I, E>(y0: State, p: LyapunovParams, integrate_fn: I) -> Result<f64, E>
where
    I: Fn([f64; 4], &[f64]) -> Result<Vec<[f64; 4]>, E>,
{
    let mut t_eval = even_grid(0.0, p.t, p.dt);
    // A horizon shorter than the grid spacing collapses `even_grid` to a single
    // entry; clamp to `[0.0, p.t]` so the integration is well-defined and the
    // short-horizon fallback below (`t_start <= 0.0` → `Ok(0.0)`) can run.
    if t_eval.len() < 2 {
        t_eval = vec![0.0, p.t];
    }
    let n_renorm = renorm_stride(p.renorm, p.dt);

    // Reference trajectory sampled on the full grid.
    let ref_sol = integrate_fn(y0.to_array(), &t_eval)?;

    // Perturbed trajectory, re-integrated from a renormalised state at every
    // checkpoint so the separation stays of order δ₀.
    let mut y_pert = y0.to_array();
    y_pert[0] += p.δ0; // perturb θ₁ by δ₀

    let half = p.t / 2.0;
    let mut log_sum = 0.0; // full sum, kept for the short-horizon fallback
    let mut tail_sum = 0.0; // sum over checkpoints in the second half
    let mut tail_start: Option<f64> = None; // first checkpoint time ≥ half
    let mut t_start = 0.0;
    let mut i = n_renorm;
    while i < t_eval.len() {
        let t_end = t_eval[i];
        let seg = integrate_fn(y_pert, &[t_start, t_end])?;
        y_pert = seg[1];

        let ref_i = ref_sol[i];
        let δ: [f64; 4] = std::array::from_fn(|k| y_pert[k] - ref_i[k]);
        let d = δ.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if d > 0.0 {
            let term = (d / p.δ0).ln();
            log_sum += term;
            if t_start >= half {
                if tail_start.is_none() {
                    tail_start = Some(t_start);
                }
                tail_sum += term;
            }
            // Renormalise: keep the direction of the deviation, reset its size.
            y_pert = std::array::from_fn(|k| (p.δ0 / d).mul_add(δ[k], ref_i[k]));
        }

        t_start = t_end;
        i += n_renorm;
    }

    // `t_start` is the last checkpoint reached, which is generally less than
    // `p.t` (the grid does not divide evenly into `renorm` steps); dividing by
    // the requested horizon would bias λ low by the uncovered tail.
    if t_start <= 0.0 {
        // The horizon is shorter than one renormalisation interval and the
        // loop body never ran; there is nothing to measure.
        return Ok(0.0);
    }
    let elapsed = tail_start.map_or(0.0, |s| t_start - s);
    Ok(if elapsed > 0.0 {
        tail_sum / elapsed
    } else {
        // Fewer than two checkpoint intervals: fall back to the plain average
        // over the single interval actually covered.
        log_sum / t_start
    })
}

/// Poincaré section: record `(θ₁, ω₁)` each time `θ₂` crosses 0 upward.
///
/// Crossings are detected on the sampled grid and refined with linear
/// interpolation. Both angles are wrapped
/// into `(−π, π]` first: the ODE integrates `θ` as an unbounded real, but the
/// section condition is a statement about the physical configuration, which
/// is 2π-periodic. Without the wrap, an orbit whose rods circulate would
/// almost never be seen to cross `θ₂ = 0` again, and the recorded `θ₁` would
/// drift by 2π every revolution.
///
/// # Errors
///
/// Returns an [`IntegratorError`] if the integration fails.
pub fn poincare_section(y0: State, p: PoincareParams) -> Result<Vec<[f64; 2]>, IntegratorError> {
    let f = |t: f64, y: &[f64; 4]| double_pendulum(t, State::from_array(*y), G).to_array();
    let t_eval = even_grid(0.0, p.t, p.dt);
    let sol = integrate(f, y0.to_array(), &t_eval, RTOL, ATOL)?;
    Ok(section_from_solution(&sol))
}

/// Extract the Poincaré section from an already-integrated solution.
///
/// Separated from [`poincare_section`] so callers with their own ODE engine
/// (the benchmark) run the identical crossing logic instead of a copy. Both
/// angles are wrapped into `(−π, π]` and the sample where the wrap jumps
/// across the branch cut is skipped.
#[must_use]
pub fn section_from_solution(sol: &[[f64; 4]]) -> Vec<[f64; 2]> {
    let mut points = Vec::new();
    for pair in sol.windows(2) {
        let (prev, cur) = (pair[0], pair[1]);
        let (p2, c2) = (wrap_pi(prev[2]), wrap_pi(cur[2]));
        // Skip the sample where the wrap itself jumps across the branch cut
        // at ±π: that is an artefact of the coordinate, not a crossing of
        // θ₂ = 0. Safe because dt is small — θ₂ cannot legitimately move by π
        // in one sample at any energy this program handles.
        if (c2 - p2).abs() > std::f64::consts::PI {
            continue;
        }
        if p2 < 0.0 && c2 >= 0.0 {
            // Linear interpolation for a slightly cleaner crossing. θ₁ is
            // interpolated before wrapping (wrapping first would blend across
            // the cut and produce garbage near ±π); ω₁ is a velocity, not an
            // angle, and is never wrapped.
            let α = -p2 / (c2 - p2);
            let θ1_cross = wrap_pi(α.mul_add(cur[0] - prev[0], prev[0]));
            let ω1_cross = α.mul_add(cur[1] - prev[1], prev[1]);
            points.push([θ1_cross, ω1_cross]);
        }
    }
    points
}

/// Wrap an angle into `(−π, π]`.
fn wrap_pi(θ: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    let w = θ.rem_euclid(TAU);
    if w > PI { w - TAU } else { w }
}

/// Number of distinct points after rounding each coordinate to `decimals`
/// places.
///
/// Kept for display purposes ([`crate::main`] uses it when printing a
/// section); the classification itself uses [`unique_scaled`], which is
/// scale-free.
#[must_use]
pub fn unique_rounded(points: &[[f64; 2]], decimals: i32) -> usize {
    let factor = 10f64.powi(decimals);
    let mut seen = HashSet::new();
    for [θ1, ω1] in points {
        // `round` first makes the value integral, so no truncation occurs, and
        // the pendulum's coordinates are bounded well within the `i64` range.
        #[allow(clippy::cast_possible_truncation)]
        let key = ((θ1 * factor).round() as i64, (ω1 * factor).round() as i64);
        seen.insert(key);
    }
    seen.len()
}

/// Number of distinct points after bucketing each coordinate at
/// `max_radius / n_buckets`, where `max_radius` is the largest distance of a
/// section point from the origin (`θ₁ = ω₁ = 0`, the stable equilibrium).
///
/// Scale-free: multiplying the whole section by a constant scales both the
/// bucket size and the section, so the count does not change. The reference
/// scale is the orbit's own size, *not* the section's span: a thin invariant
/// curve near a periodic orbit and a fat one both trace the same number of
/// span-sized cells, so span-bucketing cannot tell them apart, while relative
/// to the orbit size the thin curve collapses to a handful of cells.
///
/// # Resolution limit
///
/// Bucketing at `max_radius / n_buckets` cannot resolve an invariant curve
/// thinner than one bucket — at the shipped `n_buckets = 200` that is 1/200
/// of the orbit's own radius. Quasiperiodic orbits within roughly
/// `|ratio − √2| < 0.012` of a linear normal mode have curves below that
/// width and are reported as periodic (measured boundary: ratio 1.405,
/// Δ = 0.0092, reads periodic; ratio 1.400, Δ = 0.0142, resolves). They are
/// *nearly* periodic, so the answer is arguably right; raise `n_buckets` if
/// the distinction matters for your use, at the cost of splitting a genuinely
/// periodic section into several cells once the bucket approaches integrator
/// noise (~1e-11 relative).
#[must_use]
pub fn unique_scaled(points: &[[f64; 2]], n_buckets: f64) -> usize {
    if points.is_empty() {
        return 0;
    }
    let mut span = 0.0f64;
    let mut max_radius = 0.0f64;
    for [θ1, ω1] in points {
        max_radius = max_radius.max((θ1 * θ1 + ω1 * ω1).sqrt());
    }
    for axis in 0..2 {
        let (mn, mx) = points.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| {
            (lo.min(p[axis]), hi.max(p[axis]))
        });
        span = span.max(mx - mn);
    }
    // Collapse sections whose extent is below ~1e-9 of their own radius to a
    // single point. The threshold is relative to the orbit's size, so a tiny
    // quasiperiodic curve — which scales down with its amplitude — is never
    // mistaken for a point the way an absolute cutoff would: the thinnest
    // resolvable curve is ~0.01 of the orbit's radius, seven orders above
    // this. Genuinely periodic orbits are collapsed by the bucketing itself
    // (their extent is ~1e-5 of the radius, far below the 1/200 cell); this
    // guard only handles the machine-precision case where the section is a
    // point to ~9 decimal places of its own size.
    if span < 1e-9 * max_radius {
        return 1;
    }
    let cell = max_radius / n_buckets;
    if cell <= 0.0 {
        return 1;
    }
    let mut seen = HashSet::new();
    for [θ1, ω1] in points {
        #[allow(clippy::cast_possible_truncation)]
        seen.insert(((θ1 / cell).round() as i64, (ω1 / cell).round() as i64));
    }
    seen.len()
}

/// Full chaotic / regular classification of a starting state.
///
/// `λ_threshold` is the chaotic threshold, defaulting to 0.015: above it
/// the orbit is labelled [`Classification::Chaotic`]; below it the Poincaré
/// section decides between periodic, quasiperiodic and "need longer
/// integration".
///
/// # Errors
///
/// Returns an [`IntegratorError`] if any of the integrations fail.
pub fn classify(y0: State, λ_threshold: f64) -> Result<ClassificationResult, IntegratorError> {
    let λ = largest_lyapunov(y0, LyapunovParams::default())?;
    if λ > λ_threshold {
        return Ok(ClassificationResult {
            classification: Classification::Chaotic,
            λ,
            points: None,
        });
    }

    // Regular → examine the Poincaré section.
    let points = poincare_section(y0, PoincareParams::default())?;
    if points.len() < 10 {
        return Ok(ClassificationResult {
            classification: Classification::NeedsLongerIntegration,
            λ,
            points: Some(points),
        });
    }

    // Scale-relative uniqueness test: bucket at a fraction of the orbit's own
    // size (see [`unique_scaled`]), so amplitude does not drive the decision.
    // A periodic orbit's section collapses to a handful of cells; a
    // quasiperiodic one spreads over many. The count saturates with the
    // horizon rather than growing with it, so it must be compared against a
    // constant — comparing it against a fraction of `points.len()` makes the
    // verdict depend on the integration horizon (issue #12).
    let unique = unique_scaled(&points, 200.0);
    let classification = if unique <= 10 {
        Classification::Periodic // finite repeating set
    } else {
        Classification::Quasiperiodic // densifying a curve
    };

    Ok(ClassificationResult {
        classification,
        λ,
        points: Some(points),
    })
}

/// Number of `dt`-sized steps per renormalisation interval.
///
/// `renorm / dt` is mathematically an integer for sensible parameters, but
/// the floating-point quotient can land just below it (e.g. `0.3 / 0.1 =
/// 2.9999...`), so a plain `floor` would silently drop a whole step. Round to
/// the nearest integer instead, then clamp to at least one step.
#[must_use]
pub fn renorm_stride(renorm: f64, dt: f64) -> usize {
    // The quotient is non-negative for the positive parameters the classifier
    // uses, so the conversion cannot truncate or lose a sign.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    ((renorm / dt).round() as usize).max(1)
}

/// Evenly spaced values in `[start, stop)` with spacing `step`.
//
// Casts: `ceil` of a non-negative quotient (`start ≤ stop`, `step > 0`) and
// grid indices, both additionally capped by `take_while`, so the conversions
// cannot truncate, lose a sign, or (below 2^53 samples — 200 million years at
// the classifier's 0.02 s spacing) lose precision for the grids used here.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn even_grid(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let n = ((stop - start) / step).ceil() as usize;
    (0..n)
        .map(|i| (i as f64).mul_add(step, start))
        .take_while(|t| *t < stop)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_grid_matches_the_documented_semantics() {
        assert_eq!(even_grid(0.0, 100.0, 0.02).len(), 5000);
        assert_eq!(even_grid(0.0, 200.0, 0.01).len(), 20000);
        let t = even_grid(0.0, 100.0, 0.02);
        assert!(*t.last().unwrap() < 100.0);
        assert!(t.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn horizon_shorter_than_one_renorm_interval_returns_zero() {
        // The loop body never runs; must return Ok(0.0), not NaN or a panic.
        let p = LyapunovParams {
            t: 1.0,
            renorm: 2.0,
            ..Default::default()
        };
        let λ = largest_lyapunov(State::new(0.2, 0.0, -0.15, 0.0), p).unwrap();
        assert_eq!(λ, 0.0);
    }

    #[test]
    fn horizon_shorter_than_the_grid_spacing_returns_zero() {
        // t < dt collapses `even_grid` to a single entry; the grid must be
        // clamped to [0, t] so the short-horizon fallback runs instead of a
        // confusing InvalidTimeGrid error.
        let p = LyapunovParams {
            t: 0.005,
            dt: 0.02,
            ..Default::default()
        };
        let λ = largest_lyapunov(State::new(0.2, 0.0, -0.15, 0.0), p).unwrap();
        assert_eq!(λ, 0.0);
    }

    #[test]
    fn renorm_stride_is_not_truncated_by_float_division() {
        assert_eq!(renorm_stride(2.0, 0.02), 100);
        assert_eq!(renorm_stride(0.3, 0.1), 3); // floor() would give 2 here
        assert_eq!(renorm_stride(1.0, 0.02), 50);
        assert_eq!(renorm_stride(0.7, 0.1), 7);
        assert_eq!(renorm_stride(0.01, 1.0), 1); // clamped to at least 1
    }

    #[test]
    fn unique_rounded_collapses_nearby_points() {
        let points = [
            [0.0014, 0.2000],
            [0.0016, 0.2000],
            [0.0014, 0.2001], // duplicates the first point after rounding
            [1.0000, -2.0000],
        ];
        assert_eq!(unique_rounded(&points, 3), 3);
    }
}
