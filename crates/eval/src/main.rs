use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{self, BufWriter};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

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
    /// Collect accounts and tweets from twitterapi.io. Needs TWITTERAPI_KEY.
    Collect {
        /// The queries file: one query for each line. Lines that start with # are comments.
        #[arg(long)]
        queries: PathBuf,
        /// The output file, in JSON Lines. A run adds to it and skips the accounts it holds.
        #[arg(long)]
        out: PathBuf,
        /// Recent tweets to fetch for each account.
        #[arg(long, default_value_t = 20)]
        recent: usize,
        /// Search pages for each query. A page holds up to 20 tweets.
        #[arg(long, default_value_t = 1)]
        pages: u32,
        /// An eval set file or collected file whose accounts to skip. Repeat for more files.
        #[arg(long)]
        skip: Vec<PathBuf>,
        /// Skip the cost prompt.
        #[arg(long)]
        yes: bool,
    },
    /// Run one experiment on an eval set.
    Experiment,
    /// Write an eval report that compares experiments.
    Report,
}

fn setting(name: &'static str) -> Result<String, eval::Error> {
    std::env::var(name).map_err(|_| eval::Error::MissingSetting(name))
}

fn setting_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn read(path: &std::path::Path) -> Result<String, eval::Error> {
    std::fs::read_to_string(path).map_err(|source| eval::Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

async fn collect(
    queries: PathBuf,
    out: PathBuf,
    recent: usize,
    pages: u32,
    skip: Vec<PathBuf>,
    yes: bool,
) -> Result<ExitCode, eval::Error> {
    use eval::collect;

    let queries = collect::parse_queries(&read(&queries)?);
    if queries.is_empty() {
        return Err(eval::Error::NoQueries);
    }
    let key = setting("TWITTERAPI_KEY")?;
    let interval: u64 = setting_or("TWITTERAPI_INTERVAL_SECS", "5")
        .parse()
        .map_err(|_| eval::Error::BadSetting("TWITTERAPI_INTERVAL_SECS"))?;
    let api = scraper::TwitterApi::new(
        setting_or("TWITTERAPI_BASE_URL", "https://api.twitterapi.io"),
        key,
        pages,
        Duration::from_secs(interval),
    );

    let mut seen: HashSet<String> = HashSet::new();
    for path in &skip {
        seen.extend(collect::parse_seen_ids(&read(path)?)?);
    }
    if out.exists() {
        seen.extend(collect::parse_seen_ids(&read(&out)?)?);
    }

    let estimate = collect::estimate(queries.len(), pages, recent);
    println!(
        "{} queries x {} pages: up to {} search tweets.",
        queries.len(),
        pages,
        estimate.search_tweets
    );
    println!(
        "Up to {} accounts x {} recent tweets: up to {} recent tweets.",
        estimate.max_accounts, recent, estimate.max_recent_tweets
    );
    println!(
        "Expected cost: at most ${:.4} ({} accounts skipped already).",
        estimate.usd,
        seen.len()
    );
    let confirmed = yes
        || collect::confirm(&mut io::stdin().lock(), &mut io::stdout()).map_err(|source| {
            eval::Error::Io {
                path: "stdin".into(),
                source,
            }
        })?;
    if !confirmed {
        println!("Stopped. No call was made.");
        return Ok(ExitCode::FAILURE);
    }

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .map_err(|source| eval::Error::Io {
            path: out.clone(),
            source,
        })?;
    let summary = collect::run(&api, &queries, recent, &seen, &mut BufWriter::new(file)).await?;
    println!(
        "Collected {} accounts. Skipped {}. Failed {} accounts and {} queries.",
        summary.collected, summary.skipped, summary.failed, summary.queries_failed
    );
    println!(
        "Fetched {} search tweets and {} recent tweets. Estimated cost: ${:.4}.",
        summary.search_tweets, summary.recent_tweets, summary.usd
    );
    Ok(ExitCode::SUCCESS)
}

#[tokio::main]
async fn main() -> ExitCode {
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
        Command::Collect {
            queries,
            out,
            recent,
            pages,
            skip,
            yes,
        } => match collect(queries, out, recent, pages, skip, yes).await {
            Ok(code) => code,
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Command::Experiment | Command::Report => {
            eprintln!("not built yet");
            ExitCode::FAILURE
        }
    }
}
