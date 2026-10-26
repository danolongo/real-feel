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
    /// Join collected files and label files into an eval set file.
    Join {
        /// A collected file from `eval collect`. Repeat for more files.
        #[arg(long, required = true)]
        collected: Vec<PathBuf>,
        /// A label record file from the label agent. Repeat for more files.
        #[arg(long, required = true)]
        labels: Vec<PathBuf>,
        /// The eval set file to write.
        #[arg(long)]
        out: PathBuf,
    },
    /// Run one experiment on an eval set and store the results in Postgres.
    Experiment {
        /// The eval set file, a local copy from the eval bucket.
        #[arg(long)]
        set: PathBuf,
        #[arg(long, value_enum)]
        backend: Backend,
        #[arg(long)]
        metadata_version: String,
        #[arg(long)]
        question_version: String,
        /// How many times each item passes through the experiment.
        #[arg(long, default_value_t = 1)]
        repeats: u32,
        /// Use only the first n items, sorted by account ID.
        #[arg(long)]
        sample: Option<usize>,
        /// Skip the cost prompt (hosted backend only).
        #[arg(long)]
        yes: bool,
    },
    /// Write an eval report that compares experiments.
    Report {
        /// The Markdown file to write.
        #[arg(long)]
        out: PathBuf,
        /// An experiment ID. Repeat for more. Without it, the report covers every experiment.
        #[arg(long)]
        experiment: Vec<uuid::Uuid>,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Backend {
    Hosted,
    Fake,
}

async fn connect() -> Result<db::PgPool, eval::Error> {
    sqlx::postgres::PgPoolOptions::new()
        .acquire_timeout(Duration::from_secs(5))
        .connect(&setting("DATABASE_URL")?)
        .await
        .map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
async fn experiment(
    set: PathBuf,
    backend: Backend,
    metadata_version: String,
    question_version: String,
    repeats: u32,
    sample: Option<usize>,
    yes: bool,
) -> Result<ExitCode, eval::Error> {
    use eval::experiment::{Params, build_inputs, estimate_cost, order, run};

    let manifest = eval::manifest::from_file(&set)?;
    let items = order(eval::item::read_jsonl(&set)?, sample);
    let params = Params {
        items: &items,
        metadata_version: types::MetadataVersion(metadata_version),
        question_version: types::QuestionVersion(question_version),
        repeats,
        eval_set_sha256: manifest.sha256,
    };
    let pool = connect().await?;

    let id = match backend {
        Backend::Fake => run(&pool, &classifier::Fake::by_tweet_id(0), &params).await?,
        Backend::Hosted => {
            let inputs = build_inputs(&items, &params.metadata_version, &params.question_version)?;
            let estimate = estimate_cost(&inputs, repeats);
            println!(
                "{} items x {} repeats: about {} input tokens.",
                items.len(),
                repeats,
                estimate.input_tokens
            );
            println!(
                "Expected cost: about ${:.4} at ${} per million input tokens.",
                estimate.usd,
                eval::experiment::USD_PER_MILLION_INPUT_TOKENS
            );
            let confirmed = yes
                || eval::collect::confirm(&mut io::stdin().lock(), &mut io::stdout()).map_err(
                    |source| eval::Error::Io {
                        path: "stdin".into(),
                        source,
                    },
                )?;
            if !confirmed {
                println!("Stopped. No call was made.");
                return Ok(ExitCode::FAILURE);
            }
            run(&pool, &classifier::Hosted::from_env(), &params).await?
        }
    };

    let summary = eval::experiment::load_summary(&pool, id).await?;
    let m = &summary.metrics;
    println!("Experiment {id}");
    println!(
        "Test split at bot threshold {:.3} (set on the tune split):",
        m.threshold
    );
    println!(
        "  accuracy {:.3}  precision {:.3}  recall {:.3}  F1 {:.3}",
        m.accuracy, m.precision, m.recall, m.f1
    );
    println!(
        "  ROC-AUC {}  Brier {:.3}  ECE {:.3}",
        m.roc_auc.map_or("n/a".to_string(), |v| format!("{v:.3}")),
        m.brier,
        m.ece
    );
    println!(
        "  recall (bot) {}  recall (agent) {}",
        m.recall_bot
            .map_or("n/a".to_string(), |v| format!("{v:.3}")),
        m.recall_agent
            .map_or("n/a".to_string(), |v| format!("{v:.3}"))
    );
    Ok(ExitCode::SUCCESS)
}

async fn report(out: PathBuf, ids: Vec<uuid::Uuid>) -> Result<ExitCode, eval::Error> {
    let pool = connect().await?;
    let markdown = eval::experiment::report_markdown(&pool, &ids).await?;
    std::fs::write(&out, markdown).map_err(|source| eval::Error::Io {
        path: out.clone(),
        source,
    })?;
    println!("Wrote {}", out.display());
    Ok(ExitCode::SUCCESS)
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
        Command::Join {
            collected,
            labels,
            out,
        } => finish(join(&collected, &labels, &out)),
        Command::Experiment {
            set,
            backend,
            metadata_version,
            question_version,
            repeats,
            sample,
            yes,
        } => finish(
            experiment(
                set,
                backend,
                metadata_version,
                question_version,
                repeats,
                sample,
                yes,
            )
            .await,
        ),
        Command::Report { out, experiment } => finish(report(out, experiment).await),
    }
}

fn join(
    collected: &[PathBuf],
    labels: &[PathBuf],
    out: &std::path::Path,
) -> Result<ExitCode, eval::Error> {
    let joined = eval::item::join_files(collected, labels, out)?;
    let manifest = eval::manifest::from_file(out)?;
    println!("Wrote {} items to {}.", joined.items.len(), out.display());
    println!(
        "Dropped {} unsure and {} without a label.",
        joined.dropped_unsure, joined.dropped_unlabeled
    );
    println!(
        "Labels: {} human, {} bot, {} agent.",
        manifest.labels.human, manifest.labels.bot, manifest.labels.agent
    );
    Ok(ExitCode::SUCCESS)
}

fn finish(result: Result<ExitCode, eval::Error>) -> ExitCode {
    result.unwrap_or_else(|error| {
        eprintln!("error: {error}");
        ExitCode::FAILURE
    })
}
