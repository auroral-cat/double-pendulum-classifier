//! Command-line front end for the double-pendulum classifier.
//!
//! `cargo run` runs the two demo cases from the Python original; pass four
//! (or five, with a custom `λ_threshold`) numbers to classify your own start:
//! `cargo run -- θ1 ω1 θ2 ω2 [λ_threshold]`.

use std::process::ExitCode;

use double_pendulum_classifier::{ClassificationResult, State, classify};

/// The `λ_threshold=0.015` of the Python original.
const DEFAULT_Λ_THRESHOLD: f64 = 0.015;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.len() {
        0 => {
            run_demo();
            ExitCode::SUCCESS
        }
        4 | 5 => run_one(&args),
        _ => {
            eprintln!("usage: double-pendulum-classifier [θ1 ω1 θ2 ω2 [λ_threshold]]");
            ExitCode::from(2)
        }
    }
}

/// The two demo starts from the Python original's `__main__`.
fn run_demo() {
    println!("case 1 — small-angle start (weakly chaotic):");
    print_result(classify(
        State::new(0.2, 0.0, -0.15, 0.0),
        DEFAULT_Λ_THRESHOLD,
    ));
    println!();
    println!("case 2 — high-energy start (strongly chaotic):");
    print_result(classify(
        State::new(2.4, 0.0, 0.0, 0.0),
        DEFAULT_Λ_THRESHOLD,
    ));
    println!();
    println!("your own numbers: cargo run -- θ1 ω1 θ2 ω2 [λ_threshold]");
}

fn run_one(args: &[String]) -> ExitCode {
    let values: Result<Vec<f64>, _> = args.iter().map(|s| s.parse::<f64>()).collect();
    match values {
        Ok(v) if v.len() == 4 => {
            print_result(classify(
                State::new(v[0], v[1], v[2], v[3]),
                DEFAULT_Λ_THRESHOLD,
            ));
            ExitCode::SUCCESS
        }
        Ok(v) if v.len() == 5 => {
            print_result(classify(State::new(v[0], v[1], v[2], v[3]), v[4]));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("could not parse arguments as numbers: {args:?}");
            ExitCode::from(2)
        }
    }
}

fn print_result(result: Result<ClassificationResult, double_pendulum_classifier::IntegratorError>) {
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
        }
        Err(e) => {
            eprintln!("  integration failed: {e}");
        }
    }
}
