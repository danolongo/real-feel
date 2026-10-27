use std::process::ExitCode;

use clap::{Parser, Subcommand};
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "api-key", about = "Creates and revokes Real Feel API keys")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create an API key and print it once. Postgres stores only its hash.
    Create {
        /// The name of the key's owner.
        #[arg(long)]
        name: String,
        /// The query quota: new queries per hour.
        #[arg(long)]
        quota: i32,
    },
    /// Revoke an API key. Its requests get 401 from then on.
    Revoke {
        /// The ID that `create` printed.
        #[arg(long)]
        id: Uuid,
    },
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let url = std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL is not set")?;
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(&url)
        .await?;
    match cli.command {
        Command::Create { name, quota } => {
            if quota < 0 {
                return Err("--quota must not be negative".into());
            }
            let key = api::generate_key();
            let api_key = db::insert_api_key(&pool, &name, &api::hash_key(&key), quota).await?;
            println!("id:  {}", api_key.id);
            println!("key: {key}");
            eprintln!(
                "Save the key now. Postgres stores only its hash, so no command shows it again."
            );
        }
        Command::Revoke { id } => {
            if !db::revoke_api_key(&pool, id).await? {
                return Err(format!("no active API key has the ID {id}").into());
            }
            println!("revoked {id}");
        }
    }
    Ok(())
}
