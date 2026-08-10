//! Double-pendulum chaotic / regular classifier.
//!
//! The equations of motion, a two-trajectory (Benettin) estimate of the
//! largest Lyapunov exponent, a Poincaré section on `θ₂ = 0` crossings, and
//! the chaotic / periodic / quasiperiodic classifier.

#![forbid(unsafe_code)]

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
