//! Command-line front end for the double-pendulum classifier.
//!
//! No arguments prints the help menu (a usage error, exit 2); `-help`/`-h`
//! prints it on demand; `-demo`/`-d` runs the two built-in demo cases. Pass
//! four or five numbers — `θ1 ω1 θ2 ω2`, plus an optional `λ_threshold` — to
//! classify your own start.

use std::process::ExitCode;

use color_eyre::eyre::Report;

use double_pendulum_classifier::{ClassificationResult, IntegratorError, State, classify};

/// Default chaotic threshold: the orbit is labelled chaotic when `λ₁` exceeds
/// it.
const DEFAULT_Λ_THRESHOLD: f64 = 0.015;

/// The four state variables, in positional order.
const STATE_VARS: [&str; 4] = ["θ1", "ω1", "θ2", "ω2"];

/// The general help menu.
const HELP_TEXT: &str = "\
Double Pendulum Classifier

Classifies a double pendulum's motion as chaotic, periodic, or
quasiperiodic, given a starting state.

Usage:
    double-pendulum-classifier                  prints me
    double-pendulum-classifier -help, -h        also prints me
    double-pendulum-classifier -help [command]  prints help for a specific command
    double-pendulum-classifier -demo, -d        runs two built-in demo cases
    double-pendulum-classifier θ1 ω1 θ2 ω2 [λ_threshold]
                                                classify a given starting state

Arguments:
    θ1, θ2        initial angles, in radians
    ω1, ω2        initial angular velocities, in rad/s
    λ_threshold   chaotic threshold: orbits with λ above it are labelled
                  chaotic (optional; default 0.015)";

/// Help specific to the `-help` command.
const HELP_HELP_TEXT: &str = "\
Help for '-help':

You really have to ask?";

/// Help specific to the `-demo` command.
const HELP_DEMO_TEXT: &str = "\
Help for '-demo':

Runs the two built-in demo cases: a small-angle regular
orbit and a high-energy chaotic one.";

fn main() -> ExitCode {
    // assert: there are no other panic/error hooks
    color_eyre::install().expect("no other panic/error hook is installed");

    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = raw.iter().map(String::as_str).collect();
    match dispatch(&args) {
        Dispatch::Print(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Dispatch::Demo => run_demo(),
        Dispatch::Classify(y0, threshold) => {
            if print_result(classify(y0, threshold)) {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Dispatch::Fail(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}

/// What to do with a parsed command line.
enum Dispatch {
    /// Print the text to stdout and exit 0.
    Print(&'static str),
    /// Run the two demo cases and exit 0 or 1.
    Demo,
    /// Classify a start state and exit 0 or 1.
    Classify(State, f64),
    /// Print the message to stderr and exit 2.
    Fail(String),
}

/// Interprets the command line.
///
/// A bare invocation is a usage error: the help menu is routed through
/// `Fail` so it lands on stderr with exit 2. `-help`/`-h` accepts at most one
/// argument naming a command to explain (`help` or `demo`); anything else is
/// an unknown-command error, and extra arguments are an unknown-arguments
/// error. `-demo`/`-d` takes no arguments. The bare words `help` and `demo`
/// (no dash) are unknown commands with a did-you-mean hint.
fn dispatch(args: &[&str]) -> Dispatch {
    match args {
        [] => Dispatch::Fail(HELP_TEXT.to_string()),
        ["-help" | "-h"] => Dispatch::Print(HELP_TEXT),
        ["-help" | "-h", "help" | "-help"] => Dispatch::Print(HELP_HELP_TEXT),
        ["-help" | "-h", "demo" | "-demo"] => Dispatch::Print(HELP_DEMO_TEXT),
        ["-help" | "-h", other] => Dispatch::Fail(format!(
            "Error:\n- unknown command: '{other}' — try '-help' for the help menu.\n"
        )),
        ["-help" | "-h", ..] => Dispatch::Fail(format!(
            "Error:\n- unknown arguments: '{}' — '-help' expects at most one \
             argument: 'help' or 'demo'. Try '-help' for the help menu.\n",
            args[1..].join(" ")
        )),
        ["-demo" | "-d"] => Dispatch::Demo,
        ["-demo" | "-d", ..] => Dispatch::Fail(format!(
            "Error:\n- unknown arguments for '{}': '{}' — this command expected \
             1 argument.\n",
            args[0],
            args[1..].join(" ")
        )),
        ["help", ..] => {
            Dispatch::Fail("Error:\n- unknown command 'help' — did you mean '-help'?\n".to_string())
        }
        ["demo", ..] => {
            Dispatch::Fail("Error:\n- unknown command 'demo' — did you mean '-demo'?\n".to_string())
        }
        _ => match parse_args(args) {
            Ok((y0, threshold)) => Dispatch::Classify(y0, threshold),
            Err(report) => Dispatch::Fail(report),
        },
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

/// Parses 1–5 positional arguments (`θ1 ω1 θ2 ω2 [λ_threshold]`) into a start
/// state and chaotic threshold.
///
/// On failure — too many arguments, an argument that is not a valid float, or
/// a start state missing at least one variable — returns a preformatted report
/// naming the invalid arguments and the missing ones (`λ_threshold` is listed
/// as optional). The heading is `Error:` for a single problem and `Errors:`
/// otherwise.
fn parse_args(args: &[&str]) -> Result<(State, f64), String> {
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
        let report = parse_args(&["hi", "0.5"]).unwrap_err();
        assert_eq!(
            report,
            "Error:\n- θ1 is not a valid float.\n\nMissing Parameters:\n- θ2 ω2 [optional: λ_threshold]\n"
        );
    }

    #[test]
    fn heading_is_plural_when_more_than_one_error() {
        let report = parse_args(&["hi", "there", "0.5"]).unwrap_err();
        assert!(
            report.starts_with("Errors:\n- θ1 is not a valid float.\n- ω1 is not a valid float.\n")
        );
    }

    #[test]
    fn missing_only_when_all_given_args_are_valid() {
        let report = parse_args(&["1.0"]).unwrap_err();
        assert_eq!(
            report,
            "Missing Parameters:\n- ω1 θ2 ω2 [optional: λ_threshold]\n"
        );
    }

    #[test]
    fn three_args_miss_only_omega2_and_the_optional_threshold() {
        let report = parse_args(&["1.0", "0.0", "0.5"]).unwrap_err();
        assert_eq!(
            report,
            "Missing Parameters:\n- ω2 [optional: λ_threshold]\n"
        );
    }

    #[test]
    fn four_valid_args_parse_to_a_state_and_default_threshold() {
        let (y0, threshold) = parse_args(&["0.05", "0.0", "0.0", "0.0"]).unwrap();
        // These literals are the exact `f64` values the strings parse to.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(y0.to_array(), [0.05, 0.0, 0.0, 0.0]);
            assert_eq!(threshold, DEFAULT_Λ_THRESHOLD);
        }
    }

    #[test]
    fn five_valid_args_parse_to_a_state_and_custom_threshold() {
        let (y0, threshold) = parse_args(&["1.0", "0.0", "0.5", "0.0", "0.1"]).unwrap();
        // Same literal-exactness argument as above.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(y0.to_array(), [1.0, 0.0, 0.5, 0.0]);
            assert_eq!(threshold, 0.1);
        }
    }

    #[test]
    fn too_many_arguments_is_reported() {
        let report = parse_args(&["1.0"; 6]).unwrap_err();
        assert!(report.contains("too many arguments"));
        assert!(report.contains("got 6"));
    }

    #[test]
    fn an_invalid_fifth_argument_is_named_lambda_threshold() {
        let report = parse_args(&["1.0", "0.0", "0.5", "0.0", "hi"]).unwrap_err();
        assert!(report.contains("- λ_threshold is not a valid float.\n"));
    }

    #[test]
    fn a_bare_invocation_routes_the_help_menu_to_stderr() {
        // No arguments is a usage error: the help text goes out via `Fail`
        // (stderr, exit 2) rather than `Print` (stdout, exit 0).
        assert!(matches!(dispatch(&[]), Dispatch::Fail(_)));
    }

    #[test]
    fn help_commands_and_aliases_print_to_stdout() {
        assert!(matches!(dispatch(&["-help"]), Dispatch::Print(HELP_TEXT)));
        assert!(matches!(dispatch(&["-h"]), Dispatch::Print(HELP_TEXT)));
        assert!(matches!(
            dispatch(&["-help", "help"]),
            Dispatch::Print(HELP_HELP_TEXT)
        ));
        assert!(matches!(
            dispatch(&["-h", "demo"]),
            Dispatch::Print(HELP_DEMO_TEXT)
        ));
    }

    #[test]
    fn help_with_one_unknown_command_name_is_an_unknown_command_error() {
        let Dispatch::Fail(message) = dispatch(&["-help", "foo"]) else {
            panic!("expected Fail");
        };
        assert!(message.contains("unknown command: 'foo'"));
        assert!(message.contains("try '-help'"));
        let Dispatch::Fail(message) = dispatch(&["-h", "-demo"]) else {
            panic!("expected Fail");
        };
        assert!(message.contains("unknown command: '-demo'"));
    }

    #[test]
    fn help_with_extra_arguments_is_an_unknown_arguments_error() {
        let Dispatch::Fail(message) = dispatch(&["-help", "foo", "bar"]) else {
            panic!("expected Fail");
        };
        assert!(message.contains("unknown arguments"));
        assert!(message.contains("'foo bar'"));
        // Even a valid subcommand name followed by more arguments is an
        // unknown-arguments error, not a help page.
        let Dispatch::Fail(message) = dispatch(&["-help", "help", "x"]) else {
            panic!("expected Fail");
        };
        assert!(message.contains("unknown arguments"));
    }

    #[test]
    fn demo_commands_run_but_reject_extra_arguments() {
        assert!(matches!(dispatch(&["-demo"]), Dispatch::Demo));
        assert!(matches!(dispatch(&["-d"]), Dispatch::Demo));
        let Dispatch::Fail(message) = dispatch(&["-d", "x"]) else {
            panic!("expected Fail");
        };
        assert!(message.contains("unknown arguments for '-d'"));
        assert!(message.contains("expected 1 argument"));
    }

    #[test]
    fn bare_help_and_demo_suggest_the_dash_forms() {
        let Dispatch::Fail(message) = dispatch(&["help"]) else {
            panic!("expected Fail");
        };
        assert!(message.contains("did you mean '-help'?"));
        let Dispatch::Fail(message) = dispatch(&["demo", "x"]) else {
            panic!("expected Fail");
        };
        assert!(message.contains("did you mean '-demo'?"));
    }

    #[test]
    fn four_valid_arguments_dispatch_to_classify() {
        assert!(matches!(
            dispatch(&["0.05", "0.0", "0.0", "0.0"]),
            Dispatch::Classify(_, _)
        ));
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
