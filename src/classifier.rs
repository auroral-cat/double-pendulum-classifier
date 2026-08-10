//! Largest Lyapunov exponent (two-trajectory method), Poincaré section and the
//! chaotic / regular classifier.
//!
//! This is a faithful reimplementation of the Python experiment, except that
//! the renormalisation inside [`largest_lyapunov`] is *real*: the perturbed
//! trajectory is re-integrated from the rescaled state at every checkpoint
//! (the intended Benettin scheme). In the Python original the renormalisation
//! only edited the stored solution array after `solve_ivp` had finished, so it
//! never affected the integration and the log-sum telescoped down to a single
//! final-separation measurement.

use std::collections::HashSet;
use std::fmt;

use crate::dynamics::{G, State, double_pendulum};
use crate::integrator::{IntegratorError, integrate};

/// Solver tolerances for the integrations below.
///
/// The two-trajectory Lyapunov estimate measures the separation `d ≈ δ₀` of
/// two nearby trajectories, which is only meaningful when the integrator's
/// own trajectory error is much smaller than `δ₀ = 1e-8`. At the nominal
/// `rtol = atol = 1e-9` of the Python original the separation sits at the
/// edge of that regime and the estimate is biased high (the original itself
/// runs at the edge of its own accuracy). With `1e-12` the deviation is
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
            // noise floor that decays like 1/√T: at T = 100 it is ≈ 0.02, right
            // at the 0.015 threshold, so regular orbits read as chaotic. T =
            // 400 brings it down to ≈ 0.005, comfortably below the threshold
            // and far from the λ ≈ 1 of genuinely chaotic starts.
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
/// orbits and is `None` for chaotic ones (as in the Python original).
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
/// For a regular orbit the log-sum does not grow — it converges to a bounded
/// constant (`const ≈ 1.5` for this system), because the tangent-space
/// stretching averages out over each interval. The plain average `log_sum / t`
/// then decays like `const/t` and cannot be told apart from a genuinely small
/// growth rate: at the shipped defaults `1.5/100 = 0.015`, exactly the
/// chaotic threshold, so every regular orbit read as chaotic. The estimate
/// therefore measures the **growth** of the log-sum: only checkpoints in the
/// second half of the run contribute (`λ = tail_sum / elapsed`), so a bounded
/// log-sum yields ≈ 0 regardless of the constant while a chaotic one keeps
/// its positive growth rate.
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
pub fn largest_lyapunov_with<I, E>(y0: State, p: LyapunovParams, integrate_fn: I) -> Result<f64, E>
where
    I: Fn([f64; 4], &[f64]) -> Result<Vec<[f64; 4]>, E>,
{
    let t_eval = arange(0.0, p.t, p.dt);
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
/// interpolation, exactly as in the Python original.
///
/// # Errors
///
/// Returns an [`IntegratorError`] if the integration fails.
pub fn poincare_section(y0: State, p: PoincareParams) -> Result<Vec<[f64; 2]>, IntegratorError> {
    let f = |t: f64, y: &[f64; 4]| double_pendulum(t, State::from_array(*y), G).to_array();
    let t_eval = arange(0.0, p.t, p.dt);
    let sol = integrate(f, y0.to_array(), &t_eval, RTOL, ATOL)?;

    let mut points = Vec::new();
    for pair in sol.windows(2) {
        let (prev, cur) = (pair[0], pair[1]);
        if prev[2] < 0.0 && cur[2] >= 0.0 {
            // Linear interpolation for a slightly cleaner crossing.
            let α = -prev[2] / (cur[2] - prev[2]);
            let θ1_cross = α.mul_add(cur[0] - prev[0], prev[0]);
            let ω1_cross = α.mul_add(cur[1] - prev[1], prev[1]);
            points.push([θ1_cross, ω1_cross]);
        }
    }
    Ok(points)
}

/// Number of distinct points after rounding each coordinate to `decimals`
/// places (the `np.round(pts, decimals=3)` + `np.unique(axis=0)` test).
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

/// Full chaotic / regular classification of a starting state.
///
/// `λ_threshold` is the `λ_threshold=0.015` of the Python original: above it
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

    // Simple uniqueness test after modest rounding.
    let unique = unique_rounded(&points, 3);
    let classification = if unique < 12 {
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
pub fn renorm_stride(renorm: f64, dt: f64) -> usize {
    // The quotient is non-negative for the positive parameters the classifier
    // uses, so the conversion cannot truncate or lose a sign.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    ((renorm / dt).round() as usize).max(1)
}

/// `np.arange(start, stop, step)` — evenly spaced values in `[start, stop)`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    // `ceil` of a non-negative quotient (`start ≤ stop`, `step > 0`); the
    // grid is additionally capped by `take_while`, so the conversion cannot
    // truncate or lose a sign for the grids used here.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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
    fn arange_matches_numpy_semantics() {
        assert_eq!(arange(0.0, 100.0, 0.02).len(), 5000);
        assert_eq!(arange(0.0, 200.0, 0.01).len(), 20000);
        let t = arange(0.0, 100.0, 0.02);
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
