# Build from the repository root with .sqlx/ in the context:
#   docker build --target api -f infra/docker/Dockerfile .
#   docker build --target worker -f infra/docker/Dockerfile .
# --build-arg CARGO_PROFILE=profiling builds the images for flame graphs.
# cargo-chef compiles the dependencies in their own layer, so a code change rebuilds only the workspace crates.
FROM rust:1.98.1-bookworm AS chef
RUN cargo install cargo-chef --version 0.1.78 --locked
WORKDIR /src

FROM chef AS plan
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS build
ARG CARGO_PROFILE=release
# perf walks the call stacks through frame pointers. Only the profiling build keeps them.
ENV RUSTFLAGS_PROFILING="-C force-frame-pointers=yes"
COPY --from=plan /src/recipe.json recipe.json
RUN if [ "$CARGO_PROFILE" = profiling ]; then export RUSTFLAGS="$RUSTFLAGS_PROFILING"; fi \
    && cargo chef cook --profile "$CARGO_PROFILE" --locked --recipe-path recipe.json
COPY . .
ENV SQLX_OFFLINE=true
RUN if [ "$CARGO_PROFILE" = profiling ]; then export RUSTFLAGS="$RUSTFLAGS_PROFILING"; fi \
    && cargo build --profile "$CARGO_PROFILE" --locked -p api -p worker \
    && mkdir -p /out \
    && cp "target/$CARGO_PROFILE/api" "target/$CARGO_PROFILE/api-key" "target/$CARGO_PROFILE/worker" /out/

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home realfeel
USER realfeel

FROM runtime AS api
COPY --from=build /out/api /usr/local/bin/api
COPY --from=build /out/api-key /usr/local/bin/api-key
ENTRYPOINT ["/usr/local/bin/api"]

FROM runtime AS worker
COPY --from=build /out/worker /usr/local/bin/worker
ENTRYPOINT ["/usr/local/bin/worker"]
