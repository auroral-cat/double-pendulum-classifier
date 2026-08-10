//! End-to-end tests for the classifier and the integrator.

use double_pendulum_classifier::{
    Classification, G, PoincareParams, State, classify, double_pendulum, energy, integrate,
    poincare_section, unique_scaled,
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
fn mid_energy_start_is_quasiperiodic_under_the_default_threshold() {
    // λ for this start is ≈ 0.004 with the tail-difference estimator, well
    // below the default 0.015 threshold, so it reaches the regular branch on
    // its own — no artificially high threshold needed. It gives 73 Poincaré
    // crossings, a closed curve, so it must be labelled quasiperiodic, not
    // periodic.
    let res = classify(State::new(1.0, 0.0, 0.5, 0.0), 0.015).unwrap();
    assert_eq!(res.classification, Classification::Quasiperiodic);
    assert!(res.points.as_ref().is_some_and(|pts| pts.len() >= 10));
}

#[test]
fn equilibrium_exercises_the_needs_longer_branch() {
    // At the stable bottom equilibrium θ₂ never crosses 0, so no Poincaré
    // points are collected (the λ estimate itself is unreliable there, which
    // is why we do not assert on it).
    let res = classify(State::new(0.0, 0.0, 0.0, 0.0), 0.015).unwrap();
    assert_eq!(res.classification, Classification::NeedsLongerIntegration);
    assert_eq!(res.points.as_ref().map(Vec::len), Some(0));
}

#[test]
fn circulating_rod_two_still_crosses_the_section() {
    // Rod 2 spins over the top; its raw θ₂ runs away monotonically, but the
    // physical configuration crosses θ₂ ≡ 0 (mod 2π) dozens of times. Before
    // the wrap this returned 5 points.
    let pts = poincare_section(State::new(0.0, 0.0, 0.0, 8.0), Default::default()).unwrap();
    assert!(pts.len() >= 50, "got {} points", pts.len());
}

#[test]
fn both_rods_circulating_still_crosses_the_section() {
    // Before the wrap this returned 6 points.
    let pts = poincare_section(State::new(0.0, 3.0, 0.0, 3.0), Default::default()).unwrap();
    assert!(pts.len() >= 50, "got {} points", pts.len());
}

#[test]
fn section_angles_are_wrapped_into_minus_pi_pi() {
    let pts = poincare_section(State::new(0.0, 3.0, 0.0, 3.0), Default::default()).unwrap();
    assert!(
        pts.iter()
            .all(|[θ1, _]| *θ1 > -std::f64::consts::PI && *θ1 <= std::f64::consts::PI)
    );
}

#[test]
fn wrapping_does_not_change_a_non_circulating_section() {
    // This orbit never circulates, so wrapping must not change its section
    // (73 points, as before the fix).
    let pts = poincare_section(State::new(1.0, 0.0, 0.5, 0.0), Default::default()).unwrap();
    assert!((73..=79).contains(&pts.len()), "got {} points", pts.len());
}

#[test]
fn off_mode_orbits_classify_the_same_at_any_amplitude() {
    // ratio = 1.35 is off the √2 normal mode: genuinely quasiperiodic. The
    // classification must not depend on the amplitude (the old fixed 1e-3
    // buckets read small amplitudes as periodic and large ones as
    // quasiperiodic for the same orbit shape).
    let verdicts: Vec<_> = [0.001f64, 0.003, 0.01, 0.03, 0.1]
        .into_iter()
        .map(|a| {
            classify(State::new(a, 0.0, a * 1.35, 0.0), 0.015)
                .unwrap()
                .classification
        })
        .collect();
    assert!(
        verdicts.windows(2).all(|w| w[0] == w[1]),
        "classifications differ across amplitudes: {verdicts:?}"
    );
}

#[test]
fn classification_does_not_depend_on_the_poincare_horizon() {
    // `unique_scaled` saturates with the horizon while the point count grows,
    // so the verdict must be compared against a constant, not a fraction of
    // the point count (issue #12).
    let y0 = State::new(0.01, 0.0, 0.01 * 1.35, 0.0);
    for t in [200.0, 800.0, 3200.0] {
        let pts = poincare_section(y0, PoincareParams { t, dt: 0.01 }).unwrap();
        let u = unique_scaled(&pts, 200.0);
        assert!(u > 10, "T={t}: off-mode orbit collapsed to {u} cells");
    }
}

#[test]
fn normal_modes_are_periodic() {
    // θ₂ = ±√2·θ₁ is a normal mode of the linearised system; its section is a
    // single point (a thin torus collapses to one bucket relative to the
    // orbit size).
    let plus = classify(State::new(0.01, 0.0, 0.01 * SQRT_2, 0.0), 0.015).unwrap();
    assert_eq!(plus.classification, Classification::Periodic);
    let minus = classify(State::new(0.01, 0.0, -0.01 * SQRT_2, 0.0), 0.015).unwrap();
    assert_eq!(minus.classification, Classification::Periodic);
}

#[test]
fn off_mode_orbits_are_quasiperiodic() {
    // Same amplitude as the normal mode, different ratio: the section is a
    // real curve, not a point — the amplitude-based rule can never pass both
    // this and normal_modes_are_periodic.
    let res = classify(State::new(0.01, 0.0, 0.01 * 1.35, 0.0), 0.015).unwrap();
    assert_eq!(res.classification, Classification::Quasiperiodic);
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
