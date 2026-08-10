//! Command-line front end for the double-pendulum classifier.
//!
//! With no arguments (or `help`) the help menu is printed; `demo` runs the two
//! built-in demo cases. Pass four or five numbers — `θ1 ω1 θ2 ω2`, plus an
//! optional `λ_threshold` — to classify your own start.

use std::process::ExitCode;

use color_eyre::eyre::Report;

use double_pendulum_classifier::{ClassificationResult, IntegratorError, State, classify};

/// Default chaotic threshold: the orbit is labelled chaotic when `λ₁` exceeds
/// it.
const DEFAULT_Λ_THRESHOLD: f64 = 0.015;

/// The four state variables, in positional order.
const STATE_VARS: [&str; 4] = ["θ1", "ω1", "θ2", "ω2"];

fn main() -> ExitCode {
    // Registers color_eyre's panic/error hooks; every `Report` printed with
    // `{:?}` below renders through them (colored when stderr is a TTY).
    color_eyre::install().expect("no other panic/error hook is installed");
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => print_help(),
        [cmd] if cmd == "help" => print_help(),
        [cmd] if cmd == "demo" => run_demo(),
        _ => match parse_args(&args) {
            Ok((y0, threshold)) => {
                if print_result(classify(y0, threshold)) {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                }
            }
            Err(report) => {
                eprintln!("{report}");
                ExitCode::from(2)
            }
        },
    }
}

/// Prints the help menu. Also the default for an empty command line.
fn print_help() -> ExitCode {
    println!("Double Pendulum Classifier");
    println!();
    println!("Classifies a double pendulum's motion as chaotic, periodic, or");
    println!("quasiperiodic, given a starting state.");
    println!();
    println!("Usage:");
    println!("    double-pendulum-classifier");
    println!("    double-pendulum-classifier help");
    println!("    double-pendulum-classifier demo");
    println!("    double-pendulum-classifier θ1 ω1 θ2 ω2 [λ_threshold]");
    println!();
    println!("Commands:");
    println!("    help    Print this help menu.");
    println!("    demo    Run the two built-in demo cases (a regular and a");
    println!("            chaotic start).");
    println!();
    println!("Arguments:");
    println!("    θ1, θ2        initial angles, in radians");
    println!("    ω1, ω2        initial angular velocities, in rad/s");
    println!("    λ_threshold   chaotic threshold: orbits with λ above it are");
    println!("                  labelled chaotic (optional; default {DEFAULT_Λ_THRESHOLD})");
    println!();
    println!("Examples:");
    println!("    cargo run --release -- 0.05 0.0 0.0 0.0");
    println!("    cargo run --release -- 1.0 0.0 0.5 0.0 0.1");
    ExitCode::SUCCESS
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

/// Parses 1–5 positional arguments (`θ1 ω1 θ2 ω2 [λ_threshold]`) into a start
/// state and chaotic threshold.
///
/// On failure — too many arguments, an argument that is not a valid float, or
/// a start state missing at least one variable — returns a preformatted report
/// naming the invalid arguments and the missing ones (`λ_threshold` is listed
/// as optional). The heading is `Error:` for a single problem and `Errors:`
/// otherwise.
fn parse_args(args: &[String]) -> Result<(State, f64), String> {
    const MAX_ARGS: usize = 5;
    debug_assert!(
        !args.is_empty(),
        "main routes an empty command line to the help menu"
    );
    if args.len() > MAX_ARGS {
        return Err(format!(
            "Error:\n- too many arguments: expected at most {MAX_ARGS} \
             (θ1 ω1 θ2 ω2 [λ_threshold]), got {}",
            args.len()
        ));
    }

    let mut errors = Vec::new();
    let mut values = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        if let Ok(v) = arg.parse::<f64>() {
            values.push(v);
        } else {
            let name = STATE_VARS.get(i).copied().unwrap_or("λ_threshold");
            errors.push(format!("{name} is not a valid float."));
        }
    }

    let missing = missing_names(args.len());
    let mut report = String::new();
    if !errors.is_empty() {
        report.push_str(if errors.len() == 1 {
            "Error:\n"
        } else {
            "Errors:\n"
        });
        for error in &errors {
            report.push_str("- ");
            report.push_str(error);
            report.push('\n');
        }
        if missing.is_some() {
            report.push('\n');
        }
    }
    if let Some(missing) = missing {
        report.push_str("Missing Parameters:\n");
        report.push_str("- ");
        report.push_str(missing);
        report.push('\n');
    }
    if !errors.is_empty() || missing.is_some() {
        return Err(report);
    }

    // All four state variables are present and valid; the threshold defaults
    // when the fifth argument was not given.
    let y0 = State::new(values[0], values[1], values[2], values[3]);
    let threshold = values.get(4).copied().unwrap_or(DEFAULT_Λ_THRESHOLD);
    Ok((y0, threshold))
}

/// The state variables not covered by `n` leading arguments, followed by the
/// optional threshold. `None` when nothing required is missing.
fn missing_names(n: usize) -> Option<&'static str> {
    match n {
        1 => Some("ω1 θ2 ω2 [optional: λ_threshold]"),
        2 => Some("θ2 ω2 [optional: λ_threshold]"),
        3 => Some("ω2 [optional: λ_threshold]"),
        _ => None,
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
            let report = Report::new(e).wrap_err("integration failed");
            eprintln!("{report:?}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use double_pendulum_classifier::Classification;

    #[test]
    fn integer_strings_parse_as_floats() {
        // Rust's `f64` `FromStr` accepts "5" without a decimal point and
        // yields 5.0, so the argument parser needs no special-casing.
        assert_eq!("5".parse::<f64>(), Ok(5.0));
        assert_eq!("5.0".parse::<f64>(), Ok(5.0));
    }

    #[test]
    fn non_numeric_strings_fail_to_parse() {
        assert!("hi".parse::<f64>().is_err());
        assert!("five".parse::<f64>().is_err());
    }

    #[test]
    fn two_args_with_an_invalid_float_report_the_exact_format() {
        let report = parse_args(&["hi".to_string(), "0.5".to_string()]).unwrap_err();
        assert_eq!(
            report,
            "Error:\n- θ1 is not a valid float.\n\nMissing Parameters:\n- θ2 ω2 [optional: λ_threshold]\n"
        );
    }

    #[test]
    fn heading_is_plural_when_more_than_one_error() {
        let report =
            parse_args(&["hi".to_string(), "there".to_string(), "0.5".to_string()]).unwrap_err();
        assert!(
            report.starts_with("Errors:\n- θ1 is not a valid float.\n- ω1 is not a valid float.\n")
        );
    }

    #[test]
    fn missing_only_when_all_given_args_are_valid() {
        let report = parse_args(&["1.0".to_string()]).unwrap_err();
        assert_eq!(
            report,
            "Missing Parameters:\n- ω1 θ2 ω2 [optional: λ_threshold]\n"
        );
    }

    #[test]
    fn three_args_miss_only_omega2_and_the_optional_threshold() {
        let report =
            parse_args(&["1.0".to_string(), "0.0".to_string(), "0.5".to_string()]).unwrap_err();
        assert_eq!(
            report,
            "Missing Parameters:\n- ω2 [optional: λ_threshold]\n"
        );
    }

    #[test]
    fn four_valid_args_parse_to_a_state_and_default_threshold() {
        let (y0, threshold) = parse_args(&[
            "0.05".to_string(),
            "0.0".to_string(),
            "0.0".to_string(),
            "0.0".to_string(),
        ])
        .unwrap();
        // These literals are the exact `f64` values the strings parse to.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(y0.to_array(), [0.05, 0.0, 0.0, 0.0]);
            assert_eq!(threshold, DEFAULT_Λ_THRESHOLD);
        }
    }

    #[test]
    fn five_valid_args_parse_to_a_state_and_custom_threshold() {
        let (y0, threshold) = parse_args(&[
            "1.0".to_string(),
            "0.0".to_string(),
            "0.5".to_string(),
            "0.0".to_string(),
            "0.1".to_string(),
        ])
        .unwrap();
        // Same literal-exactness argument as above.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(y0.to_array(), [1.0, 0.0, 0.5, 0.0]);
            assert_eq!(threshold, 0.1);
        }
    }

    #[test]
    fn too_many_arguments_is_reported() {
        let report = parse_args(&vec!["1.0".to_string(); 6]).unwrap_err();
        assert!(report.contains("too many arguments"));
        assert!(report.contains("got 6"));
    }

    #[test]
    fn an_invalid_fifth_argument_is_named_lambda_threshold() {
        let report = parse_args(&[
            "1.0".to_string(),
            "0.0".to_string(),
            "0.5".to_string(),
            "0.0".to_string(),
            "hi".to_string(),
        ])
        .unwrap_err();
        assert!(report.contains("- λ_threshold is not a valid float.\n"));
    }

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
