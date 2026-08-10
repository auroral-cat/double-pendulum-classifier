# Double Pendulum Classifier

A small Rust program that watches a double pendulum and tells you
what kind of motion it is in: **chaotic**, **periodic**, or **quasiperiodic**.

## AI Usage Policy

Most or all of this code was written by `DeepSeek-V4-Flash-0731` via the `Pi` coding harness. 
`GLM-5.2` (via `Pi`) & `Claude-Opus-5` (via `Claude Code`) were used for code review. 
As such, please use at your own risk. This project was written to sate personal curiosity.

This usage policy was also the only thing fully written by a human being. If nothing else, 
this is probably the only thing that "must" be written by hand nowadays to maintain respect for you, the reader.

## The problem

The double pendulum is one of the best-known examples of chaos.
Depending on where you start it, it can:

1. Swing gently in a repeating rhythm (periodic),
2. Drift through a pattern that never quite repeats (quasiperiodic),
3. Behave so unpredictably that nearby starting points lead to wildly
  different futures (chaotic).

*So given a starting state, which one will it be?*
You cannot tell just by looking — the answer depends on the exact starting
state in a surprisingly complicated way. This program answers the question
automatically.

## How it works

The program takes a starting state — the two angles `θ1`, `θ2` and the two
angular velocities `ω1`, `ω2` — and runs three steps:

1. **Simulate the motion.** A built-in Dormand–Prince (RK45) integrator steps
   the equations of motion forward in time, with per-component error control
   and accuracy guarantees you would expect from a scientific library.

2. **Measure the chaos with a Lyapunov exponent.** Two nearly identical
   trajectories are followed side by side. If they drift apart exponentially,
   the motion is chaotic; the growth rate is the largest Lyapunov exponent
   `λ`. The separation is periodically renormalized (the standard Benettin
   method) so the measurement stays accurate even over long integrations.

3. **For regular motion, look at a Poincaré section.** If `λ` is small, the
   program records where the trajectory crosses a fixed slice of the motion
   (each time `θ2` passes through 0 going upward). A finite set of crossing
   points means periodic; points that fill in a curve mean quasiperiodic.

The classification rule is: `λ` above a threshold (default 0.015) means chaotic;
otherwise the Poincaré section decides whether it's quasiperiodic or periodic.

## Getting started

This requires Rust 2024 edition. The crate uses minimal dependencies for error reporting;
the solver itself is pure standard library.

```bash
# Classify your own starting state: θ1 ω1 θ2 ω2
cargo run --release -- 0.05 0.0 0.0 0.0

# You can also adjust the chaotic threshold: θ1 ω1 θ2 ω2 λ_threshold
cargo run --release -- 1.0 0.0 0.5 0.0 0.1

# Run the two built-in demo cases
cargo run --release -- -demo

# Bring up the help menu
cargo run --release -- -help

# Run the test suite
cargo test
```

Example output:

```
case 1 — small-angle start (regular):
  λ = 0.0051  →  quasiperiodic (177 Poincaré points, 177 unique after rounding to 3 decimals)

case 2 — high-energy start (strongly chaotic):
  λ = 1.1352  →  chaotic
```

## Project layout

- `src/dynamics.rs` — the equations of motion and the energy function
- `src/integrator.rs` — the adaptive RK45 integrator
- `src/classifier.rs` — Lyapunov exponent, Poincaré section, classification
- `src/main.rs` — the CLI front end
- `tests/classification.rs` — end-to-end tests
- `benches/compare.rs` — a criterion benchmark comparing this integrator with
  popular ODE crates

## Performance

A benchmark against established ODE crates (`ode_solvers`, `diffsol`,
`peroxide`) is included. Against the two adaptive solvers — `ode_solvers`'
Dopri5 and `diffsol`'s TSIT45 — the built-in integrator is the fastest by
about 1.4–1.9× at `rtol = atol = 1e-12` (measured on the author's machine;
the exact margin depends on the hardware).

The `peroxide` row is **not** a like-for-like comparison: peroxide's adaptive
API cannot land on requested output times, so it is driven as fixed-step RK4
at `dt = 5e-4`. That does considerably more work than an adaptive method
needs, and the ~4.6× gap reflects the step size rather than the crate. See
the comments in `benches/compare.rs` for details.

Run it yourself with:

```bash
cargo bench --bench compare
```

## Notes
- A word of caution: no classifier is perfect. The Lyapunov estimate has a
  small noise floor, so the threshold is not a sharp physical boundary —
  orbits with `λ` very close to it are borderline by construction. If a
  result seems surprising, try a longer integration or a different starting
  state.

## Contributing
Open an issue or submit a PR; though this is a personal project, conversation is welcome!
