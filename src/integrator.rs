//! A self-contained Dormand–Prince 5(4) (RK45) adaptive integrator.
//!
//! Per-component `rtol`/`atol` scaling with an RMS error norm, free adaptive
//! steps chosen solely by the error controller, and dense output: the
//! requested output times are read from the Dormand–Prince interpolant, so
//! the results are independent of the output grid. Each step is bounded by
//! the end of the requested span; it is *not* clipped at the individual
//! output times, which would pin the step size to the grid spacing.

use std::error::Error;
use std::fmt;

/// Stage times `c_s` of the Dormand–Prince 5(4) pair.
const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

/// Runge–Kutta coefficients `a_{sj}` (`j < s`); entries with `j ≥ s` are 0.
const A: [[f64; 7]; 7] = [
    [0.0; 7],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ],
];

/// Fifth-order weights `b_s`.
const B5: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];

/// Dense-output coefficients `P` (the Shampine quartic interpolant for the
/// Dormand–Prince pair):
///
/// ```text
/// y(t_old + x·h) = y_old + h·Σ_s k_s·(P[s][0]·x + P[s][1]·x² + P[s][2]·x³ + P[s][3]·x⁴)
/// ```
///
/// Literature constants (Hairer et al., *Solving Ordinary Differential
/// Equations I*, Table 5.3) pasted verbatim: the raw integer form is the
/// canonical representation, so the literals are intentionally left without
/// digit separators.
#[allow(clippy::unreadable_literal)]
const P: [[f64; 4]; 7] = [
    [
        1.0,
        -8048581381.0 / 2820520608.0,
        8663915743.0 / 2820520608.0,
        -12715105075.0 / 11282082432.0,
    ],
    [0.0; 4],
    [
        0.0,
        131558114200.0 / 32700410799.0,
        -68118460800.0 / 10900136933.0,
        87487479700.0 / 32700410799.0,
    ],
    [
        0.0,
        -1754552775.0 / 470086768.0,
        14199869525.0 / 1410260304.0,
        -10690763975.0 / 1880347072.0,
    ],
    [
        0.0,
        127303824393.0 / 49829197408.0,
        -318862633887.0 / 49829197408.0,
        701980252875.0 / 199316789632.0,
    ],
    [
        0.0,
        -282668133.0 / 205662961.0,
        2019193451.0 / 616988883.0,
        -1453857185.0 / 822651844.0,
    ],
    [
        0.0,
        40617522.0 / 29380423.0,
        -110615467.0 / 29380423.0,
        69997945.0 / 29380423.0,
    ],
];

/// Fourth-order (embedded) weights `b*_s`; the local error estimate is
/// `h · Σ_s (b_s − b*_s) k_s`. Same literature provenance as `P` — the raw
/// integer form is the canonical representation.
#[allow(clippy::unreadable_literal)]
const B4: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

/// Failure modes of [`integrate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntegratorError {
    /// The output grid must have at least two strictly increasing entries.
    InvalidTimeGrid,
    /// `rtol` and `atol` must both be strictly positive.
    InvalidTolerance { rtol: f64, atol: f64 },
    /// The adaptive step size collapsed below the ulp of `t` at time `t`
    /// (the right-hand side is probably stiff or singular).
    StepSizeUnderflow { t: f64 },
}

impl fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeGrid => {
                write!(
                    f,
                    "t_eval must have at least two strictly increasing entries"
                )
            }
            Self::InvalidTolerance { rtol, atol } => {
                write!(
                    f,
                    "rtol and atol must be strictly positive (got {rtol} and {atol})"
                )
            }
            Self::StepSizeUnderflow { t } => write!(f, "step size underflow at t = {t}"),
        }
    }
}

impl Error for IntegratorError {}

/// Integrate `ẏ = f(t, y)` starting from `y0` at `t_eval[0]`, sampling the
/// solution at every entry of `t_eval`.
///
/// `t_eval` must be strictly increasing. The Dormand–Prince 5(4) pair runs
/// with free adaptive steps chosen solely by the error controller, and each
/// requested output time is read from the step's dense-output interpolant —
/// the step sequence does not depend on the output grid. `rtol` and `atol`
/// are the per-component relative/absolute tolerances and must both be
/// strictly positive.
///
/// # Errors
///
/// - [`IntegratorError::InvalidTimeGrid`] if `t_eval` has fewer than two
///   entries or is not strictly increasing.
/// - [`IntegratorError::InvalidTolerance`] if `rtol` or `atol` is not
///   strictly positive.
/// - [`IntegratorError::StepSizeUnderflow`] if the adaptive step size
///   collapses, e.g. for a stiff or singular right-hand side.
pub fn integrate<const N: usize, F>(
    f: F,
    y0: [f64; N],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
) -> Result<Vec<[f64; N]>, IntegratorError>
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
{
    if t_eval.len() < 2 || t_eval.windows(2).any(|w| w[1] <= w[0]) {
        return Err(IntegratorError::InvalidTimeGrid);
    }
    if !(rtol > 0.0 && atol > 0.0) {
        return Err(IntegratorError::InvalidTolerance { rtol, atol });
    }

    let mut ys = Vec::with_capacity(t_eval.len());
    ys.push(y0);

    let mut y = y0;
    let (mut h, f0) = select_initial_step(&f, t_eval[0], &y0, rtol, atol);
    // Dormand–Prince is a FSAL (first-same-as-last) pair: the final stage
    // `k₇ = f(t + h, y₅)` of an accepted step is exactly the first stage of
    // the next step, so it only has to be evaluated once per step. The initial
    // step's first stage comes free from `select_initial_step`, which already
    // evaluated `f(t₀, y₀)` to pick `h`.
    let mut k0 = Some(f0);
    let mut t = t_eval[0];
    let mut next = 1; // index of the next `t_eval` entry to output
    // `t_eval` is validated to have at least two strictly increasing entries,
    // so `t_eval[t_eval.len() - 1]` is the end of the requested span.
    let t_end = t_eval[t_eval.len() - 1];

    // The step sequence is independent of the output grid: each accepted step
    // advances by the controller's own `h`, and the requested output points
    // are read from the dense-output interpolant. Clipping the step at every
    // `t_eval` entry (as this module once did) pins the step size to the grid
    // spacing and makes the step sequence a function of the grid, which
    // injects spurious separation into the two-trajectory Lyapunov
    // measurement: the same trajectory integrated over a fine and a coarse
    // grid took different step sequences and diverged by ~1e-10 at t = 50 s.
    loop {
        // Bound the step by the end of the requested span. Clipping here does
        // *not* reintroduce the output-grid dependence that dense output
        // removed (issue #7):
        // only the final step is affected, and `t_end` is part of the
        // caller's request, not of how densely the span is sampled.
        let h_step = h.min(t_end - t);
        // The step must be large enough to actually advance `t`; otherwise
        // `t += h_step` is a no-op and the loop would livelock (accepted steps
        // make no progress, then `h` grows and is rejected again).
        if t + h_step <= t {
            return Err(IntegratorError::StepSizeUnderflow { t });
        }
        let step = rk45_step(&f, t, &y, h_step, rtol, atol, k0);
        // The controller's next proposal, from the error of the step actually
        // taken (`h_step`, possibly clipped at `t_end`). The `h.max` keeps the
        // previous estimate when the step was clipped and accepted: a
        // boundary clip is not an error signal, so it must not shrink `h`.
        let proposed = step_factor(step.err_norm) * h_step;
        h = if step.err_norm <= 1.0 && h_step < h {
            h.max(proposed)
        } else {
            proposed
        };
        if step.err_norm <= 1.0 {
            // Collect every output point covered by this step via the
            // interpolant (using the step's own size, not the next proposal).
            while next < t_eval.len() && t_eval[next] <= t + h_step {
                let x = (t_eval[next] - t) / h_step;
                ys.push(dense_output(&y, h_step, &step.k, x));
                next += 1;
            }
            t += h_step;
            y = step.y5;
            // `k₇ = f(t + h, y₅)` is `f` at the new step origin; reuse it as
            // the next step's first stage. On rejection the step origin is
            // unchanged, so the previous `k0` stays valid.
            k0 = Some(step.k[6]);
            if next == t_eval.len() {
                break;
            }
        }
    }

    Ok(ys)
}

/// One Dormand–Prince step of size `h`.
struct DpStep<const N: usize> {
    /// 5th-order update `y₅`.
    y5: [f64; N],
    /// RMS error norm with per-component scaling
    /// `atol + rtol · max(|y|, |y₅|)`.
    err_norm: f64,
    /// All seven stages `k_s = f(t + c_s·h, y_s)`, kept for the dense-output
    /// interpolant and the FSAL reuse (`k[6]` is `f` at the new step origin).
    k: [[f64; N]; 7],
}

/// One Dormand–Prince step of size `h`: returns the 5th-order update, the
/// RMS error norm, and the seven stages.
///
/// `k0` is the first stage `f(t, y)`, usually reused from the previous
/// accepted step (see [`integrate`]); pass `None` to evaluate it fresh.
fn rk45_step<const N: usize, F>(
    f: &F,
    t: f64,
    y: &[f64; N],
    h: f64,
    rtol: f64,
    atol: f64,
    k0: Option<[f64; N]>,
) -> DpStep<N>
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
{
    let mut k = [[0.0; N]; 7];
    k[0] = k0.unwrap_or_else(|| f(t, y));

    for (s, c_s) in C.iter().enumerate().skip(1) {
        let mut stage = *y;
        for (a_sj, k_j) in A[s].iter().zip(&k) {
            for (stage_n, k_jn) in stage.iter_mut().zip(k_j) {
                *stage_n += h * a_sj * k_jn;
            }
        }
        k[s] = f(t + c_s * h, &stage);
    }

    let mut y5 = *y;
    let mut y4 = *y;
    for ((b5_s, b4_s), k_s) in B5.iter().zip(&B4).zip(&k) {
        for ((y5_n, y4_n), k_sn) in y5.iter_mut().zip(&mut y4).zip(k_s) {
            *y5_n += h * b5_s * k_sn;
            *y4_n += h * b4_s * k_sn;
        }
    }

    let err_sq = y5
        .iter()
        .zip(&y4)
        .zip(y)
        .map(|((y5_n, y4_n), y_n)| {
            let scale = rtol.mul_add(y_n.abs().max(y5_n.abs()), atol);
            let e = (y5_n - y4_n) / scale;
            e * e
        })
        .sum::<f64>();
    // `N` is a const-generic state dimension, always 4 in this codebase;
    // a dimension needing >= 2^53 components would lose precision.
    #[allow(clippy::cast_precision_loss)]
    let err_norm = (err_sq / N as f64).sqrt();

    DpStep { y5, err_norm, k }
}

/// Evaluate the dense-output interpolant of an accepted step at
/// `x = (t − t_old)/h ∈ [0, 1]` (see the `P` coefficients above).
fn dense_output<const N: usize>(y_old: &[f64; N], h: f64, k: &[[f64; N]; 7], x: f64) -> [f64; N] {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let mut out = *y_old;
    for n in 0..N {
        let mut sum = 0.0;
        for (k_s, p_s) in k.iter().zip(&P) {
            sum += k_s[n] * (p_s[0] * x + p_s[1] * x2 + p_s[2] * x3 + p_s[3] * x4);
        }
        out[n] += h * sum;
    }
    out
}

/// Step-size factor for the error norm:
/// `clamp(0.9 · err^(−0.2), 0.2, 10)`. A non-finite error norm is treated as a
/// rejection (`0.2`) so the loop shrinks the step instead of poisoning it.
fn step_factor(err_norm: f64) -> f64 {
    if err_norm.is_nan() {
        0.2
    } else {
        (0.9 * err_norm.powf(-0.2)).clamp(0.2, 10.0)
    }
}

/// Initial step size from the first stage of Hairer's heuristic (Hairer
/// et al., *Solving Ordinary Differential Equations I*, eq. 4.14):
/// `h = 0.01 · ‖y₀‖/‖f(t₀, y₀)‖` in the `rtol`/`atol`-scaled norm. The second
/// stage of the heuristic (a trial Euler step to estimate the local
/// curvature) is omitted — the adaptive controller corrects any overestimate
/// within the first few steps.
///
/// Returns the evaluation `f0 = f(t0, y0)` alongside the step size so the
/// caller can seed the first step's first stage; `f0` is exactly `k₁` of a
/// step starting at `(t0, y0)`, so reusing it saves one right-hand-side
/// evaluation per [`integrate`] call.
fn select_initial_step<const N: usize, F>(
    f: &F,
    t0: f64,
    y0: &[f64; N],
    rtol: f64,
    atol: f64,
) -> (f64, [f64; N])
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
{
    let f0 = f(t0, y0);
    let (mut sum_y2, mut sum_f2) = (0.0, 0.0);
    for (y_n, f_n) in y0.iter().zip(&f0) {
        let scale = rtol.mul_add(y_n.abs(), atol);
        let y_scaled = y_n / scale;
        let f_scaled = f_n / scale;
        sum_y2 = y_scaled.mul_add(y_scaled, sum_y2);
        sum_f2 = f_scaled.mul_add(f_scaled, sum_f2);
    }
    // Same reasoning as in `rk45_step`: `N` is a small const dimension.
    #[allow(clippy::cast_precision_loss)]
    let d0 = (sum_y2 / N as f64).sqrt();
    #[allow(clippy::cast_precision_loss)]
    let d1 = (sum_f2 / N as f64).sqrt();
    let h = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    (h, f0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_growth_matches_analytics() {
        let f = |_t: f64, y: &[f64; 1]| [y[0]];
        let t_eval = [0.0, 0.5, 1.0, 1.5, 2.0];
        let sol = integrate(f, [1.0], &t_eval, 1e-9, 1e-9).unwrap();
        for (t, y) in t_eval.iter().zip(&sol) {
            let expected = t.exp();
            assert!(
                (y[0] - expected).abs() < 1e-7,
                "t = {t}: {} vs {expected}",
                y[0]
            );
        }
    }

    #[test]
    fn rejects_non_increasing_time_grid() {
        let f = |_t: f64, y: &[f64; 1]| [y[0]];
        let t_eval = [0.0, 1.0, 1.0];
        assert_eq!(
            integrate(f, [1.0], &t_eval, 1e-9, 1e-9),
            Err(IntegratorError::InvalidTimeGrid)
        );
    }

    #[test]
    fn rejects_non_positive_tolerances() {
        let f = |_t: f64, y: &[f64; 1]| [y[0]];
        let t_eval = [0.0, 1.0];
        assert_eq!(
            integrate(f, [1.0], &t_eval, 0.0, 1e-9),
            Err(IntegratorError::InvalidTolerance {
                rtol: 0.0,
                atol: 1e-9
            })
        );
        assert!(matches!(
            integrate(f, [1.0], &t_eval, 1e-9, -1.0),
            Err(IntegratorError::InvalidTolerance {
                rtol: 1e-9,
                atol: -1.0
            })
        ));
    }

    #[test]
    fn blow_up_reports_underflow_instead_of_hanging() {
        let f = |_t: f64, y: &[f64; 1]| [y[0] * y[0] * y[0]];
        // y' = y³ with y(0) = 10 blows up at t = 1/(2·10²) = 0.005; must
        // return an error, not spin forever.
        let r = integrate(f, [10.0], &[0.0, 10.0], 1e-9, 1e-9);
        assert!(
            matches!(r, Err(IntegratorError::StepSizeUnderflow { t }) if (t - 0.005).abs() < 1e-4),
            "expected StepSizeUnderflow near t = 0.005, got {r:?}"
        );
    }

    #[test]
    fn initial_step_seeds_the_first_stage_without_an_extra_evaluation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let f = |_t: f64, y: &[f64; 1]| {
            calls.fetch_add(1, Ordering::Relaxed);
            [y[0]]
        };
        let sol = integrate(f, [1.0], &[0.0, 2.0], 1e-9, 1e-9).unwrap();
        assert!((sol[1][0] - 2.0f64.exp()).abs() < 1e-7);
        // `select_initial_step`'s f0 is reused as the first step's first
        // stage, so the total is 1 + 6 per `rk45_step` (six new stages per
        // step). Without the seed it would be 2 + 6 per step.
        let c = calls.load(Ordering::Relaxed);
        assert_eq!(c % 6, 1, "expected 1 + 6·steps RHS calls, got {c}");
    }

    #[test]
    fn the_step_never_runs_past_the_requested_span() {
        use std::cell::Cell;
        let max_t = Cell::new(f64::MIN);
        let f = |t: f64, y: &[f64; 1]| {
            max_t.set(max_t.get().max(t));
            [-0.001 * y[0]]
        };
        let sol = integrate(f, [1.0], &[0.0, 1.0], 1e-6, 1e-6).unwrap();
        assert!(
            max_t.get() <= 1.0,
            "rhs evaluated at t = {} > 1.0",
            max_t.get()
        );
        assert!((sol[1][0] - (-0.001f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn the_step_sequence_stays_independent_of_the_output_grid() {
        // Issue #7: the same trajectory on a fine and a coarse grid must agree
        // exactly at the shared times. Clipping at `t_end` (issue #15) must not
        // break this. The coarse grid is a *subset* of the fine one, so the
        // shared times are bit-identical — comparing two independently
        // computed grids (e.g. `i * 0.02` vs `i * 0.2`) would inject a 1-ulp
        // difference in the evaluation time itself (round-to-even tie-break at
        // 0.6, 1.2) and fail on a pure representation artifact.
        let f = |_t: f64, y: &[f64; 1]| [y[0]];
        let fine: Vec<f64> = (0..=100).map(|i| f64::from(i) * 0.02).collect();
        let coarse: Vec<f64> = fine.iter().step_by(10).copied().collect();
        let sol_fine = integrate(f, [1.0], &fine, 1e-12, 1e-12).unwrap();
        let sol_coarse = integrate(f, [1.0], &coarse, 1e-12, 1e-12).unwrap();
        for k in 1..coarse.len() {
            // Bit-identicality is the property under test (issues #7, #15):
            // any 1-ulp divergence is a step-sequence regression.
            #[allow(clippy::float_cmp)]
            {
                assert_eq!(
                    sol_fine[k * 10][0],
                    sol_coarse[k][0],
                    "mismatch at t = {}",
                    coarse[k]
                );
            }
        }
    }
}
