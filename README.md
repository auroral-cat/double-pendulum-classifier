# Double Pendulum Classifier

A small, friendly Rust program that watches a double pendulum and tells you
what kind of motion it is in: **chaotic**, **periodic**, or **quasiperiodic**.

## The problem

The double pendulum — two rods connected end to end, swinging under gravity —
is one of the best-known examples of chaos. Depending on where you start it,
it can:

- swing gently in a repeating rhythm (periodic),
- drift through a pattern that never quite repeats (quasiperiodic),
- or behave so unpredictably that nearby starting points lead to wildly
  different futures (chaotic).

The interesting question is: *given a starting state, which one will it be?*
You cannot tell just by looking — the answer depends on the exact starting
state in a surprisingly complicated way. This program answers the question
automatically.

## How it works

The program takes a starting state — the two angles `θ1`, `θ2` and the two
angular velocities `ω1`, `ω2` — and runs three steps:

1. **Simulate the motion.** A built-in Dormand–Prince (RK45) integrator steps
   the equations of motion forward in time, with per-component error control
   and the same accuracy guarantees you would get from a scientific library.

2. **Measure the chaos with a Lyapunov exponent.** Two nearly identical
   trajectories are followed side by side. If they drift apart exponentially,
   the motion is chaotic; the growth rate is the largest Lyapunov exponent
   `λ`. The separation is periodically renormalized (the standard Benettin
   method) so the measurement stays accurate even over long integrations.

3. **For regular motion, look at a Poincaré section.** If `λ` is small, the
   program records where the trajectory crosses a fixed slice of the motion
   (each time `θ2` passes through 0 going upward). A finite set of crossing
   points means periodic; points that fill in a curve mean quasiperiodic.

The classification rule is simple: `λ` above a threshold (default 0.015)
means chaotic; otherwise the Poincaré section decides.

## Getting started

You need a recent Rust toolchain (edition 2024). There are **no external
dependencies** — it builds with just the standard library.

```bash
# Run the two built-in demo cases
cargo run --release

# Classify your own starting state: θ1 ω1 θ2 ω2
cargo run --release -- 0.05 0.0 0.0 0.0

# You can also adjust the chaotic threshold: θ1 ω1 θ2 ω2 λ_threshold
cargo run --release -- 1.0 0.0 0.5 0.0 0.1

# Run the test suite
cargo test

# Check the code quality
cargo clippy --all-targets
```

Example output:

```
case 1 — small-angle start (regular):
  λ = 0.0028  →  quasiperiodic (177 Poincaré points, 177 unique after rounding to 3 decimals)

case 2 — high-energy start (strongly chaotic):
  λ = 1.1087  →  chaotic
```

## Project layout

- `src/dynamics.rs` — the equations of motion and the energy function
- `src/integrator.rs` — the adaptive RK45 integrator
- `src/classifier.rs` — Lyapunov exponent, Poincaré section, classification
- `src/main.rs` — the command-line front end
- `tests/classification.rs` — end-to-end tests
- `benches/compare.rs` — a criterion benchmark comparing this integrator with
  popular ODE crates

## Performance

This Rust version is fast: the two demo cases finish in about 25 milliseconds.

A benchmark against established ODE crates (`ode_solvers`, `diffsol`,
`peroxide`) shows the built-in integrator is the fastest of the four on this
problem, by about 1.2–1.5× over the closest contenders and 4× over
`peroxide`. Run it yourself with:

```bash
cargo bench --bench compare
```

## Notes

- The code uses Greek-letter identifiers (`θ1`, `ω1`, `λ`, `δ0`, …), which
  Rust fully supports.
- The binary ships with zero dependencies; the benchmark crates are
  dev-dependencies only.
- A word of caution: no classifier is perfect. The Lyapunov estimate has a
  small noise floor, so the threshold is not a sharp physical boundary —
  orbits with `λ` very close to it are borderline by construction. If a
  result seems surprising, try a longer integration or a different starting
  state.
