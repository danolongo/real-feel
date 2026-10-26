# Build from the repository root with .sqlx/ in the context:
#   docker build --target api -f infra/docker/Dockerfile .
#   docker build --target worker -f infra/docker/Dockerfile .
# cargo-chef compiles the dependencies in their own layer, so a code change rebuilds only the workspace crates.
FROM rust:1.98.1-bookworm AS chef
RUN cargo install cargo-chef --version 0.1.78 --locked
WORKDIR /src

FROM chef AS plan
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS build
COPY --from=plan /src/recipe.json recipe.json
RUN cargo chef cook --release --locked --recipe-path recipe.json
COPY . .
ENV SQLX_OFFLINE=true
RUN cargo build --release --locked -p api -p worker

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home realfeel
USER realfeel

FROM runtime AS api
COPY --from=build /src/target/release/api /usr/local/bin/api
ENTRYPOINT ["/usr/local/bin/api"]

FROM runtime AS worker
COPY --from=build /src/target/release/worker /usr/local/bin/worker
ENTRYPOINT ["/usr/local/bin/worker"]
