//! Double-pendulum chaotic / regular classifier.
//!
//! The equations of motion, a two-trajectory (Benettin) estimate of the
//! largest Lyapunov exponent, a Poincaré section on `θ₂ = 0` crossings, and
//! the chaotic / periodic / quasiperiodic classifier.

#![forbid(unsafe_code)]
// Allow `similar_names` and `many_single_char_names`: these pedantic lints
// fire on deliberately short, standard names — `stop`/`step` for the grid
// helpers (the NumPy/Matlab arange convention), `sum_y2`/`sum_f2` (Hairer's
// notation), and the ODE symbols `f, t, y, h, k, x` — where renaming would
// make the math harder to read. Crate-root attributes are used so that
// `-W clippy::pedantic` on the command line honours the allowance.
#![allow(clippy::similar_names, clippy::many_single_char_names)]

pub mod classifier;
pub mod dynamics;
pub mod integrator;

pub use classifier::{
    Classification, ClassificationResult, LyapunovParams, PoincareParams, classify,
    largest_lyapunov, largest_lyapunov_with, poincare_section, renorm_stride,
    section_from_solution, unique_rounded, unique_scaled,
};
pub use dynamics::{G, State, double_pendulum, energy};
pub use integrator::{IntegratorError, integrate};
