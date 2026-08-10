//! End-to-end tests for the classifier and the integrator.

use double_pendulum_classifier::{
    Classification, G, State, classify, double_pendulum, energy, integrate,
};

const SQRT_2: f64 = std::f64::consts::SQRT_2;

#[test]
fn tiny_linear_normal_mode_is_not_chaotic() {
    // A start at 1e-4 rad is a linear harmonic oscillator to eight decimal
    // places; its true λ₁ is 0, so it must not read as chaotic.
    let res = classify(State::new(1e-4, 0.0, 1e-4 * SQRT_2, 0.0), 0.015).unwrap();
    assert_ne!(res.classification, Classification::Chaotic);
}

#[test]
fn tiny_generic_start_is_not_chaotic() {
    // Deep in the KAM regime: bounded motion, λ₁ = 0.
    let res = classify(State::new(1e-3, 0.0, 5e-4, 0.0), 0.015).unwrap();
    assert_ne!(res.classification, Classification::Chaotic);
}

#[test]
fn high_energy_start_is_strongly_chaotic() {
    let res = classify(State::new(2.4, 0.0, 0.0, 0.0), 0.015).unwrap();
    assert_eq!(res.classification, Classification::Chaotic);
    // The literature value is λ₁ ≈ 1.09; a correct Benettin estimate must
    // land well above 0.5 (the buggy Python original reported only ≈ 0.22).
    assert!(
        res.λ > 0.5,
        "λ = {} should be a large positive exponent",
        res.λ
    );
}

#[test]
fn small_angle_start_is_not_chaotic() {
    // E = −2g·cos(0.2) − g·cos(−0.15) = −28.93, only ~1.7% above the potential
    // floor −3g = −29.43: deep in the KAM regime, where invariant tori
    // dominate. The old 1/T-normalised estimate mislabelled this orbit
    // "weakly chaotic" — see issue #1.
    let res = classify(State::new(0.2, 0.0, -0.15, 0.0), 0.015).unwrap();
    assert_ne!(res.classification, Classification::Chaotic);
    assert!(res.λ.abs() < 0.01, "λ = {} should be near zero", res.λ);
}

#[test]
fn high_threshold_exercises_the_quasiperiodic_branch() {
    // With an artificially high threshold the regular branch runs; this start
    // gives 73 Poincaré crossings (Python: 73 unique after rounding), so it
    // must be labelled quasiperiodic, not periodic.
    let res = classify(State::new(1.0, 0.0, 0.5, 0.0), 2.0).unwrap();
    assert_eq!(res.classification, Classification::Quasiperiodic);
    assert!(res.points.as_ref().is_some_and(|pts| pts.len() >= 10));
}

#[test]
fn equilibrium_exercises_the_needs_longer_branch() {
    // At the stable bottom equilibrium θ₂ never crosses 0, so no Poincaré
    // points are collected (the λ estimate itself is unreliable there, which
    // is why we do not assert on it).
    let res = classify(State::new(0.0, 0.0, 0.0, 0.0), 2.0).unwrap();
    assert_eq!(res.classification, Classification::NeedsLongerIntegration);
    assert_eq!(res.points.as_ref().map(Vec::len), Some(0));
}

#[test]
fn energy_is_conserved_over_the_integration() {
    let y0 = State::new(0.2, 0.0, -0.15, 0.0);
    let t_eval: Vec<f64> = (0..=200).map(|i| i as f64 * 0.1).collect(); // 0 s → 20 s
    let f = |t: f64, y: &[f64; 4]| double_pendulum(t, State::from_array(*y), G).to_array();
    let sol = integrate(f, y0.to_array(), &t_eval, 1e-9, 1e-9).unwrap();

    let e0 = energy(y0, G);
    for (t, y) in t_eval.iter().zip(&sol) {
        let e = energy(State::from_array(*y), G);
        let drift = ((e - e0) / e0).abs();
        assert!(drift < 1e-6, "relative energy drift at t = {t}: {drift:e}");
    }
}
