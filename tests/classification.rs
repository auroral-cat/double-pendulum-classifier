//! End-to-end tests for the classifier and the integrator.

use double_pendulum_classifier::{
    Classification, G, State, classify, double_pendulum, energy, integrate,
};

#[test]
fn small_angle_start_is_weakly_chaotic() {
    let res = classify(State::new(0.2, 0.0, -0.15, 0.0), 0.015).unwrap();
    assert_eq!(res.classification, Classification::Chaotic);
    assert!(
        res.λ > 0.02,
        "λ = {} should be clearly above the threshold",
        res.λ
    );
}

#[test]
fn high_energy_start_is_strongly_chaotic() {
    let res = classify(State::new(2.4, 0.0, 0.0, 0.0), 0.015).unwrap();
    assert_eq!(res.classification, Classification::Chaotic);
    // The true λ₁ is ≈ 1.1; a correct Benettin estimate must land well above 0.5
    // (the buggy Python original reported only ≈ 0.22 here).
    assert!(
        res.λ > 0.5,
        "λ = {} should be a large positive exponent",
        res.λ
    );
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
