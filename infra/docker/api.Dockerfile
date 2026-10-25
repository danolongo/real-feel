# Build from the repository root with .sqlx/ in the context:
#   docker build --platform linux/arm64 -f infra/docker/api.Dockerfile .
FROM rust:1.98.1-bookworm AS build
WORKDIR /src
COPY . .
ENV SQLX_OFFLINE=true
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p api \
    && cp target/release/api /usr/local/bin/api

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home realfeel
COPY --from=build /usr/local/bin/api /usr/local/bin/api
USER realfeel
ENTRYPOINT ["/usr/local/bin/api"]
