//! Command-line front end for the double-pendulum classifier.
//!
//! `cargo run` runs two demo cases; pass four (or five, with a custom
//! `λ_threshold`) numbers to classify your own start:
//! `cargo run -- θ1 ω1 θ2 ω2 [λ_threshold]`.

use std::process::ExitCode;

use double_pendulum_classifier::{ClassificationResult, IntegratorError, State, classify};

/// Default chaotic threshold: the orbit is labelled chaotic when `λ₁` exceeds
/// it.
const DEFAULT_Λ_THRESHOLD: f64 = 0.015;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.len() {
        0 => run_demo(),
        4 | 5 => run_one(&args),
        _ => {
            eprintln!("usage: double-pendulum-classifier [θ1 ω1 θ2 ω2 [λ_threshold]]");
            ExitCode::from(2)
        }
    }
}

/// The two demo starts: a small-angle regular orbit and a high-energy
/// chaotic one.
fn run_demo() -> ExitCode {
    println!("case 1 — small-angle start (regular):");
    let case1 = print_result(classify(
        State::new(0.2, 0.0, -0.15, 0.0),
        DEFAULT_Λ_THRESHOLD,
    ));
    println!();
    println!("case 2 — high-energy start (strongly chaotic):");
    let case2 = print_result(classify(
        State::new(2.4, 0.0, 0.0, 0.0),
        DEFAULT_Λ_THRESHOLD,
    ));
    println!();
    println!("your own numbers: cargo run -- θ1 ω1 θ2 ω2 [λ_threshold]");
    if case1 && case2 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn run_one(args: &[String]) -> ExitCode {
    let values: Result<Vec<f64>, _> = args.iter().map(|s| s.parse::<f64>()).collect();
    let (y0, threshold) = match values {
        Ok(v) if v.len() == 4 => (State::new(v[0], v[1], v[2], v[3]), DEFAULT_Λ_THRESHOLD),
        Ok(v) if v.len() == 5 => (State::new(v[0], v[1], v[2], v[3]), v[4]),
        _ => {
            eprintln!("could not parse arguments as numbers: {args:?}");
            return ExitCode::from(2);
        }
    };
    if print_result(classify(y0, threshold)) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Prints one classification; returns `true` when it succeeded.
fn print_result(result: Result<ClassificationResult, IntegratorError>) -> bool {
    match result {
        Ok(res) => {
            print!("  λ = {:.4}  →  {}", res.λ, res.classification);
            if let Some(points) = &res.points {
                let unique = double_pendulum_classifier::unique_rounded(points, 3);
                println!(
                    " ({} Poincaré points, {unique} unique after rounding to 3 decimals)",
                    points.len()
                );
            } else {
                println!();
            }
            true
        }
        Err(e) => {
            eprintln!("  integration failed: {e}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use double_pendulum_classifier::Classification;

    #[test]
    fn failed_classification_is_reported_as_failure() {
        let err = IntegratorError::StepSizeUnderflow { t: 0.5 };
        assert!(!print_result(Err(err)));
    }

    #[test]
    fn successful_classification_is_reported_as_success() {
        let ok = ClassificationResult {
            classification: Classification::Chaotic,
            λ: 0.5,
            points: None,
        };
        assert!(print_result(Ok(ok)));
    }
}
