use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "eval",
    about = "Builds, runs, and reports Real Feel experiments"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the sha256, item count, and label counts of an eval set file.
    Manifest {
        /// The eval set file, in JSON Lines.
        eval_set: PathBuf,
    },
    /// Collect accounts and tweets from twitterapi.io.
    Collect,
    /// Run one experiment on an eval set.
    Experiment,
    /// Write an eval report that compares experiments.
    Report,
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Manifest { eval_set } => match eval::manifest::from_file(&eval_set)
            .and_then(|manifest| Ok(serde_json::to_string_pretty(&manifest)?))
        {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Command::Collect | Command::Experiment | Command::Report => {
            eprintln!("not built yet");
            ExitCode::FAILURE
        }
    }
}
