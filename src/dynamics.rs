//! Equations of motion for the double pendulum.
//!
//! Equal masses `m₁ = m₂ = 1` and equal lengths `L₁ = L₂ = 1`; the state
//! vector is `y = [θ₁, ω₁, θ₂, ω₂]` (angles in radians, angular velocities in
//! rad/s), where `θ₁` and `θ₂` are the angles of the first and second rods
//! measured from the downward vertical (both absolute, not relative to
//! each other).

/// Gravitational acceleration in m/s² (the `g` default of the Python original).
pub const G: f64 = 9.81;

/// Full state of the double pendulum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    pub θ1: f64,
    pub ω1: f64,
    pub θ2: f64,
    pub ω2: f64,
}

impl State {
    pub const fn new(θ1: f64, ω1: f64, θ2: f64, ω2: f64) -> Self {
        Self { θ1, ω1, θ2, ω2 }
    }

    pub const fn to_array(self) -> [f64; 4] {
        [self.θ1, self.ω1, self.θ2, self.ω2]
    }

    pub const fn from_array(y: [f64; 4]) -> Self {
        Self {
            θ1: y[0],
            ω1: y[1],
            θ2: y[2],
            ω2: y[3],
        }
    }
}

/// Right-hand side `ẏ = f(t, y)` of the double-pendulum ODE.
///
/// The equations are kept in their textbook form for readability; clippy's
/// `suboptimal_flops` would rewrite every product as a fused multiply–add
/// chain, which would obscure the mathematics without changing the result to
/// leading order.
#[allow(clippy::suboptimal_flops)]
pub fn double_pendulum(_t: f64, y: State, g: f64) -> State {
    let State { θ1, ω1, θ2, ω2 } = y;
    let c = (θ1 - θ2).cos();
    let s = (θ1 - θ2).sin();
    let den = s.mul_add(s, 1.0); // m₁ = m₂ = 1

    let ω1_dot = (g * θ2.sin() * c - s * (ω1 * ω1 * c + ω2 * ω2) - 2.0 * g * θ1.sin()) / den;
    let ω2_dot = (2.0 * (ω1 * ω1 * s - g * θ2.sin() + g * θ1.sin() * c) + ω2 * ω2 * s * c) / den;

    State {
        θ1: ω1,
        ω1: ω1_dot,
        θ2: ω2,
        ω2: ω2_dot,
    }
}

/// Total mechanical energy `E = T + V` for `m = L = 1`.
///
/// Useful as a conservation check for the integrator. Kept in textbook form
/// for the same readability reason as [`double_pendulum`].
#[allow(clippy::suboptimal_flops)]
pub fn energy(y: State, g: f64) -> f64 {
    let State { θ1, ω1, θ2, ω2 } = y;
    let kinetic = ω1 * ω1 + 0.5 * ω2 * ω2 + ω1 * ω2 * (θ1 - θ2).cos();
    let potential = -g * (2.0 * θ1.cos() + θ2.cos());
    kinetic + potential
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_equilibrium_is_a_fixed_point() {
        let y = State::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(double_pendulum(0.0, y, G), y);
    }

    #[test]
    fn energy_at_the_bottom_is_minus_three_g() {
        let e = energy(State::new(0.0, 0.0, 0.0, 0.0), G);
        let expected = -3.0 * G;
        assert!((e - expected).abs() < 1e-12);
    }
}
