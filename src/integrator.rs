//! A self-contained Dormand–Prince 5(4) (RK45) adaptive integrator.
//!
//! It reproduces the error-control semantics of scipy's `solve_ivp` with
//! `method="RK45"` — per-component `rtol`/`atol` scaling, an RMS error norm
//! and step clipping at the requested output times — so that results stay
//! comparable with the original Python experiment.

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

/// Fourth-order (embedded) weights `b*_s`; the local error estimate is
/// `h · Σ_s (b_s − b*_s) k_s`.
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
    /// The adaptive step size collapsed below `f64::MIN_POSITIVE` at time `t`
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
            Self::StepSizeUnderflow { t } => write!(f, "step size underflow at t = {t}"),
        }
    }
}

impl Error for IntegratorError {}

/// Integrate `ẏ = f(t, y)` starting from `y0` at `t_eval[0]`, sampling the
/// solution at every entry of `t_eval`.
///
/// `t_eval` must be strictly increasing; each interval is integrated with the
/// Dormand–Prince 5(4) pair, clipping the adaptive step at the interval end
/// (the same strategy `solve_ivp` uses for `t_eval` output). `rtol` and
/// `atol` are the per-component relative/absolute tolerances and must both be
/// positive.
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
    assert!(rtol > 0.0 && atol > 0.0, "rtol and atol must be positive");

    let mut ys = Vec::with_capacity(t_eval.len());
    ys.push(y0);

    let mut y = y0;
    let mut h = select_initial_step(&f, t_eval[0], &y0, rtol, atol);

    for interval in t_eval.windows(2) {
        let (ta, tb) = (interval[0], interval[1]);
        let mut t = ta;
        while t < tb {
            let h_step = h.min(tb - t);
            if h_step < f64::MIN_POSITIVE {
                return Err(IntegratorError::StepSizeUnderflow { t });
            }
            let (y_new, err_norm) = rk45_step(&f, t, &y, h_step, rtol, atol);
            h = step_factor(err_norm) * h_step;
            if err_norm <= 1.0 {
                t += h_step;
                y = y_new;
            }
        }
        ys.push(y);
    }

    Ok(ys)
}

/// One Dormand–Prince step of size `h`: returns the 5th-order update and the
/// RMS error norm with per-component scaling `atol + rtol · max(|y|, |y₅|)`.
fn rk45_step<const N: usize, F>(
    f: &F,
    t: f64,
    y: &[f64; N],
    h: f64,
    rtol: f64,
    atol: f64,
) -> ([f64; N], f64)
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
{
    let mut k = [[0.0; N]; 7];
    k[0] = f(t, y);

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
            let scale = atol + rtol * y_n.abs().max(y5_n.abs());
            let e = (y5_n - y4_n) / scale;
            e * e
        })
        .sum::<f64>();
    let err_norm = (err_sq / N as f64).sqrt();

    (y5, err_norm)
}

/// Step-size factor for the error norm, scipy-style:
/// `clamp(0.9 · err^(−0.2), 0.2, 10)`. A non-finite error norm is treated as a
/// rejection (`0.2`) so the loop shrinks the step instead of poisoning it.
fn step_factor(err_norm: f64) -> f64 {
    if err_norm.is_nan() {
        0.2
    } else {
        (0.9 * err_norm.powf(-0.2)).clamp(0.2, 10.0)
    }
}

/// Initial step size following scipy's `select_initial_step` heuristic
/// (Hairer et al., *Solving Ordinary Differential Equations I*, eq. 4.14).
fn select_initial_step<const N: usize, F>(
    f: &F,
    t0: f64,
    y0: &[f64; N],
    rtol: f64,
    atol: f64,
) -> f64
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
{
    let f0 = f(t0, y0);
    let (mut sum_y2, mut sum_f2) = (0.0, 0.0);
    for (y_n, f_n) in y0.iter().zip(&f0) {
        let scale = atol + rtol * y_n.abs();
        let y_scaled = y_n / scale;
        let f_scaled = f_n / scale;
        sum_y2 += y_scaled * y_scaled;
        sum_f2 += f_scaled * f_scaled;
    }
    let d0 = (sum_y2 / N as f64).sqrt();
    let d1 = (sum_f2 / N as f64).sqrt();
    if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    }
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
}
