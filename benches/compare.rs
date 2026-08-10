//! Backend comparison benchmark: hand-rolled RK45 vs `ode_solvers`,
//! `diffsol` and `peroxide`.
//!
//! Every backend runs the *same* workloads — a 400 s reference integration,
//! the full Benettin Lyapunov estimate (reference + 199 renormalised segment
//! restarts) and the 200 s Poincaré section — at `rtol = atol = 1e-12`, the
//! tolerance shipped in the classifier. The Lyapunov and Poincaré control
//! flow is shared (the library's own [`largest_lyapunov_with`]); only the ODE
//! engine differs.
//!
//! A verification table (λ values, crossing counts) is printed before the
//! timing runs so that the results can be checked for scientific equivalence.
//! It is *not* bit-level agreement: the `ode_solvers` backend snaps to the
//! nearest stored dense-output sample via `nearest_index`, so two backends
//! may legitimately differ in the last few digits of `y_end`.
//!
//! [`largest_lyapunov_with`]: double_pendulum_classifier::largest_lyapunov_with

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};

use double_pendulum_classifier::{LyapunovParams, State};

use diffsol::{DenseMatrix, NalgebraMat, OdeBuilder, OdeSolverMethod};
use ode_solvers::{Dopri5, OutputType, System, Vector4};
use peroxide::fuga::{ODEIntegrator, ODEProblem, RK4};

const G: f64 = 9.81;
const RTOL: f64 = 1e-12;
const ATOL: f64 = 1e-12;

const CASE1: [f64; 4] = [0.2, 0.0, -0.15, 0.0];
const CASE2: [f64; 4] = [2.4, 0.0, 0.0, 0.0];
const POINCARE_Y0: [f64; 4] = [1.0, 0.0, 0.5, 0.0];
/// Rod 2 spins over the top: raw θ₂ runs away monotonically, so a crossing
/// test on unwrapped θ₂ reports ~5 points while the wrapped library logic
/// reports dozens. Keeps the verification table honest about the wrap.
const CIRCULATING_Y0: [f64; 4] = [0.0, 0.0, 0.0, 8.0];

/// Right-hand side of the double pendulum on a plain `[f64; 4]`, shared by all
/// backends so that the equations are bit-identical. Kept in textbook form,
/// matching `dynamics::double_pendulum`.
#[allow(clippy::suboptimal_flops)]
fn pendulum_rhs(y: [f64; 4]) -> [f64; 4] {
    let θ1 = y[0];
    let ω1 = y[1];
    let θ2 = y[2];
    let ω2 = y[3];
    let c = (θ1 - θ2).cos();
    let s = (θ1 - θ2).sin();
    let den = s.mul_add(s, 1.0);
    let ω1_dot = (G * θ2.sin() * c - s * (ω1 * ω1 * c + ω2 * ω2) - 2.0 * G * θ1.sin()) / den;
    let ω2_dot = (2.0 * (ω1 * ω1 * s - G * θ2.sin() + G * θ1.sin() * c) + ω2 * ω2 * s * c) / den;
    [ω1, ω1_dot, ω2, ω2_dot]
}

/// `np.arange(start, stop, step)`.
fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    // See `classifier::arange` for why this conversion is safe.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = ((stop - start) / step).ceil() as usize;
    (0..n)
        .map(|i| (i as f64).mul_add(step, start))
        .take_while(|t| *t < stop)
        .collect()
}

/// Index of the element of the sorted `xs` closest to `t`.
fn nearest_index(xs: &[f64], t: f64) -> usize {
    let i = xs.partition_point(|x| *x < t);
    if i == 0 {
        0
    } else if i == xs.len() {
        xs.len() - 1
    } else if t - xs[i - 1] < xs[i] - t {
        i - 1
    } else {
        i
    }
}

/// A pluggable ODE engine: integrate `y0` from `t_eval[0]` to
/// `t_eval.last()`, returning the solution sampled at every `t_eval` entry.
trait Backend {
    fn name(&self) -> &'static str;
    fn integrate(&self, y0: [f64; 4], t_eval: &[f64]) -> Result<Vec<[f64; 4]>, String>;
}

// ---------------------------------------------------------------------------
// Hand-rolled Dormand–Prince RK45 (the shipped integrator)
// ---------------------------------------------------------------------------
struct HandRolled;

impl Backend for HandRolled {
    fn name(&self) -> &'static str {
        "hand_rolled_rk45"
    }

    fn integrate(&self, y0: [f64; 4], t_eval: &[f64]) -> Result<Vec<[f64; 4]>, String> {
        let f = |_t: f64, y: &[f64; 4]| pendulum_rhs(*y);
        double_pendulum_classifier::integrator::integrate(f, y0, t_eval, RTOL, ATOL)
            .map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// ode_solvers 0.6.2 — Dormand–Prince Dopri5 (dense output at fixed dx)
// ---------------------------------------------------------------------------
struct Pendulum;

impl System<f64, Vector4<f64>> for Pendulum {
    fn system(&self, _t: f64, y: &Vector4<f64>, dy: &mut Vector4<f64>) {
        let r = pendulum_rhs([y[0], y[1], y[2], y[3]]);
        *dy = Vector4::new(r[0], r[1], r[2], r[3]);
    }
}

struct OdeSolversBackend;

impl Backend for OdeSolversBackend {
    fn name(&self) -> &'static str {
        "ode_solvers_dopri5"
    }

    fn integrate(&self, y0: [f64; 4], t_eval: &[f64]) -> Result<Vec<[f64; 4]>, String> {
        let x0 = t_eval[0];
        let x_end = *t_eval.last().unwrap();
        // For a grid, dense output at the grid spacing; for a 2-point segment,
        // sparse output (the last accepted step lands exactly on x_end).
        let dx = if t_eval.len() > 2 {
            t_eval[1] - t_eval[0]
        } else {
            x_end - x0
        };
        let out_type = if t_eval.len() == 2 {
            OutputType::Sparse
        } else {
            OutputType::Dense
        };
        let mut solver = Dopri5::from_param(
            Pendulum,
            x0,
            x_end,
            dx,
            Vector4::new(y0[0], y0[1], y0[2], y0[3]),
            RTOL,
            ATOL,
            0.9,        // safety factor
            0.04,       // PI-controller beta
            0.2,        // fac_min
            10.0,       // fac_max
            x_end - x0, // h_max
            0.0,        // h (0 = auto)
            // The default 100_001-step cap is too small for the 400 s
            // Lyapunov reference run (~120k steps at rtol = 1e-12).
            200_000,
            1000, // n_stiff
            out_type,
        );
        solver.integrate().map_err(|e| e.to_string())?;
        let xs = solver.x_out();
        let ys = solver.y_out();
        Ok(t_eval
            .iter()
            .map(|&t| {
                let y = &ys[nearest_index(xs, t)];
                [y[0], y[1], y[2], y[3]]
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// diffsol 0.16.1 — explicit TSIT45 Runge–Kutta, dense output at t_eval
// ---------------------------------------------------------------------------
struct DiffsolBackend;

impl Backend for DiffsolBackend {
    fn name(&self) -> &'static str {
        "diffsol_tsit45"
    }

    fn integrate(&self, y0: [f64; 4], t_eval: &[f64]) -> Result<Vec<[f64; 4]>, String> {
        type M = NalgebraMat<f64>;
        let problem = OdeBuilder::<M>::new()
            .rtol(RTOL)
            .atol([ATOL; 4])
            .t0(t_eval[0])
            .rhs(|x, _p, _t, y| {
                let r = pendulum_rhs([x[0], x[1], x[2], x[3]]);
                y[0] = r[0];
                y[1] = r[1];
                y[2] = r[2];
                y[3] = r[3];
            })
            .init(
                move |_p, _t, y| {
                    y[0] = y0[0];
                    y[1] = y0[1];
                    y[2] = y0[2];
                    y[3] = y0[3];
                },
                4,
            )
            .build()
            .map_err(|e| e.to_string())?;
        let mut solver = problem.tsit45().map_err(|e| e.to_string())?;
        let (ys, _stop) = solver.solve_dense(t_eval).map_err(|e| e.to_string())?;
        Ok((0..t_eval.len())
            .map(|i| {
                let col = ys.column(i);
                [col[0], col[1], col[2], col[3]]
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// peroxide 0.43.1 — RK4 with a fixed-step driver.
//
// peroxide's adaptive API (`BasicODESolver`/`RKF45`) outputs at its own
// adaptive steps and cannot land exactly on requested times (its `step`
// method does not expose the accepted step size), which is required by the
// Benettin checkpoint restarts. Its fixed-step RK4 accepts every step, so it
// is driven here with a step size small enough to match the reference
// accuracy (verified against the other backends below).
// ---------------------------------------------------------------------------
struct PendulumProblem;

impl ODEProblem for PendulumProblem {
    fn rhs(&self, _t: f64, y: &[f64], dy: &mut [f64]) -> anyhow::Result<()> {
        let r = pendulum_rhs([y[0], y[1], y[2], y[3]]);
        dy.copy_from_slice(&r);
        Ok(())
    }
}

struct PeroxideBackend {
    /// Fixed RK4 step size, chosen so the deviation δ₀ = 1e-8 stays resolved.
    dt: f64,
}

impl Backend for PeroxideBackend {
    fn name(&self) -> &'static str {
        "peroxide_rk4"
    }

    fn integrate(&self, y0: [f64; 4], t_eval: &[f64]) -> Result<Vec<[f64; 4]>, String> {
        let t_end = *t_eval.last().unwrap();
        let mut y = y0;
        let mut t = t_eval[0];
        let mut out = Vec::with_capacity(t_eval.len());
        let mut next = 0;
        let rk4 = RK4;
        let problem = PendulumProblem;
        // `loop`/`break` form of the floating-point step loop: RK4 advances by
        // exactly `h`, so `t` reaches `t_end` after a finite number of steps.
        loop {
            let h = self.dt.min(t_end - t);
            rk4.step(&problem, t, &mut y, h)
                .map_err(|e| e.to_string())?;
            t += h;
            // Collect any grid points the step has reached or passed.
            while next < t_eval.len() {
                if t < t_eval[next] - 1e-9 {
                    break;
                }
                out.push(y);
                next += 1;
            }
            if t >= t_end {
                break;
            }
        }
        while next < t_eval.len() {
            out.push(y);
            next += 1;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Shared workloads
// ---------------------------------------------------------------------------

/// Full Benettin Lyapunov estimate — runs the library's own algorithm
/// ([`largest_lyapunov_with`](double_pendulum_classifier::largest_lyapunov_with))
/// through the given backend, so the benchmark measures the identical control
/// flow on every engine instead of maintaining a drift-prone copy.
fn lyapunov_workload(b: &dyn Backend, y0: [f64; 4]) -> f64 {
    double_pendulum_classifier::largest_lyapunov_with(
        State::from_array(y0),
        LyapunovParams::default(),
        |y, t_eval| b.integrate(y, t_eval),
    )
    .expect("lyapunov integration")
}

/// Poincaré section: count upward crossings of `θ₂ = 0` over 200 s.
/// Poincaré section: count crossings over 200 s, using the library's own
/// crossing logic so the benchmark cannot drift from it (issues #10, #16).
fn poincare_workload(b: &dyn Backend, y0: [f64; 4]) -> usize {
    let t_eval = arange(0.0, 200.0, 0.01);
    let sol = b.integrate(y0, &t_eval).expect("poincaré integration");
    double_pendulum_classifier::section_from_solution(&sol).len()
}

/// Single 100 s reference integration sampled at dt = 0.02.
fn integrate_workload(b: &dyn Backend) -> [f64; 4] {
    let t_eval = arange(0.0, 100.0, 0.02);
    let sol = b.integrate(CASE1, &t_eval).expect("integration");
    *sol.last().unwrap()
}

fn backends() -> Vec<Box<dyn Backend>> {
    vec![
        Box::new(HandRolled),
        Box::new(OdeSolversBackend),
        Box::new(DiffsolBackend),
        Box::new(PeroxideBackend { dt: 5e-4 }),
    ]
}

/// Check that every backend computes the same physics before timing it.
fn verify() {
    println!("verification (rtol = atol = {:e}):", RTOL);
    println!(
        "{:<20} {:>10} {:>10} {:>14} {:>14} {:>12}",
        "backend", "λ case1", "λ case2", "poincaré pts", "circulating", "y_end[0]"
    );
    for b in backends() {
        let λ1 = lyapunov_workload(b.as_ref(), CASE1);
        let λ2 = lyapunov_workload(b.as_ref(), CASE2);
        let n = poincare_workload(b.as_ref(), POINCARE_Y0);
        let n_circ = poincare_workload(b.as_ref(), CIRCULATING_Y0);
        let ye = integrate_workload(b.as_ref());
        println!(
            "{:<20} {:>10.5} {:>10.5} {:>14} {:>14} {:>12.8}",
            b.name(),
            λ1,
            λ2,
            n,
            n_circ,
            ye[0]
        );
    }
    println!();
}

fn configure() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(30)
}

fn bench_integrate(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrate_100s");
    for b in backends() {
        group.bench_function(BenchmarkId::new("integrate", b.name()), |bb| {
            bb.iter(|| black_box(integrate_workload(b.as_ref())))
        });
    }
    group.finish();
}

fn bench_lyapunov(c: &mut Criterion) {
    let mut group = c.benchmark_group("largest_lyapunov");
    for b in backends() {
        group.bench_function(BenchmarkId::new("lyapunov", b.name()), |bb| {
            bb.iter(|| black_box(lyapunov_workload(b.as_ref(), CASE1)))
        });
    }
    group.finish();
}

fn bench_poincare(c: &mut Criterion) {
    let mut group = c.benchmark_group("poincare_section");
    for b in backends() {
        group.bench_function(BenchmarkId::new("poincare", b.name()), |bb| {
            bb.iter(|| black_box(poincare_workload(b.as_ref(), POINCARE_Y0)))
        });
    }
    group.finish();
}

/// Run every benchmark group and print criterion's end-of-run summary.
fn benches() {
    let mut criterion = configure().configure_from_args();
    bench_integrate(&mut criterion);
    bench_lyapunov(&mut criterion);
    bench_poincare(&mut criterion);
    // With `harness = false`, criterion's end-of-run summary (and the
    // change-vs-last-run comparison) is only printed if asked for explicitly.
    criterion.final_summary();
}

fn main() {
    verify();
    if std::env::args().any(|a| a == "--verify-only") {
        return;
    }
    benches();
}
