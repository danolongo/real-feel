use std::collections::HashMap;

use opentelemetry::{
    propagation::TextMapPropagator,
    trace::{TraceContextExt, TracerProvider as _},
};
use opentelemetry_otlp::{ExporterBuildError, SpanExporter};
use opentelemetry_sdk::{Resource, propagation::TraceContextPropagator, trace::SdkTracerProvider};
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

const TRACEPARENT: &str = "traceparent";
/// Real Feel's crates log at `info`. Their dependencies log only warnings.
const DEFAULT_FILTER: &str =
    "warn,api=info,worker=info,scraper=info,classifier=info,queue=info,db=info";

/// Flushes the pending spans when it drops. Keep it alive until the binary exits.
pub struct Guard(Option<SdkTracerProvider>);

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(provider) = &self.0 {
            let _ = provider.shutdown();
        }
    }
}

/// Installs JSON logs, and a trace exporter when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
/// `service` names the process in the trace viewer.
pub fn init(service: &str) -> Result<Guard, ExporterBuildError> {
    let provider = if std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some() {
        let exporter = SpanExporter::builder().with_http().build()?;
        Some(
            SdkTracerProvider::builder()
                .with_resource(
                    Resource::builder()
                        .with_service_name(service.to_string())
                        .build(),
                )
                .with_batch_exporter(exporter)
                .build(),
        )
    } else {
        None
    };
    let otel_layer = provider
        .as_ref()
        .map(|provider| tracing_opentelemetry::layer().with_tracer(provider.tracer("real-feel")));

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| DEFAULT_FILTER.into()))
        .with(tracing_subscriber::fmt::layer().json())
        .with(otel_layer)
        .init();
    Ok(Guard(provider))
}

/// Returns the trace ID of `span`, or `None` when no trace is active.
pub fn trace_id(span: &Span) -> Option<String> {
    let context = span.context();
    let span_context = context.span().span_context().clone();
    span_context
        .is_valid()
        .then(|| span_context.trace_id().to_string())
}

/// Waits for SIGTERM or Ctrl-C. A binary that returns after it drops the guard and flushes the spans.
pub async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("cannot listen for SIGTERM");
    tokio::select! {
        _ = terminate.recv() => {}
        _ = tokio::signal::ctrl_c() => {}
    }
}

/// Returns the current span's context as a W3C `traceparent` string.
/// Returns `None` when no trace is active.
pub fn current_traceparent() -> Option<String> {
    let mut carrier = HashMap::new();
    TraceContextPropagator::new().inject_context(&Span::current().context(), &mut carrier);
    carrier.remove(TRACEPARENT)
}

/// Makes `span` a child of the span that `traceparent` names.
pub fn continue_trace(span: &Span, traceparent: &str) {
    let carrier = HashMap::from([(TRACEPARENT.to_string(), traceparent.to_string())]);
    let context = TraceContextPropagator::new().extract(&carrier);
    let _ = span.set_parent(context);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_tracing<T>(test: impl FnOnce() -> T) -> T {
        let provider = SdkTracerProvider::builder().build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
        tracing::subscriber::with_default(subscriber, test)
    }

    fn trace_id(traceparent: &str) -> &str {
        traceparent.split('-').nth(1).unwrap()
    }

    #[test]
    fn traceparent_is_none_without_a_trace() {
        assert_eq!(current_traceparent(), None);
    }

    #[test]
    fn continued_span_keeps_the_trace_id() {
        with_tracing(|| {
            let sent = tracing::info_span!("sender").in_scope(|| current_traceparent().unwrap());

            let receiver = tracing::info_span!("receiver");
            continue_trace(&receiver, &sent);
            let received = receiver.in_scope(|| current_traceparent().unwrap());

            assert_eq!(trace_id(&received), trace_id(&sent));
            assert_ne!(received, sent);
        });
    }

    #[test]
    fn trace_id_names_the_continued_trace() {
        with_tracing(|| {
            let sent = tracing::info_span!("sender").in_scope(|| current_traceparent().unwrap());
            let receiver = tracing::info_span!("receiver");
            continue_trace(&receiver, &sent);
            assert_eq!(super::trace_id(&receiver).unwrap(), trace_id(&sent));
        });
    }

    #[test]
    fn trace_id_is_none_without_a_trace() {
        assert_eq!(super::trace_id(&tracing::info_span!("alone")), None);
    }

    #[test]
    fn unrelated_span_gets_a_new_trace_id() {
        with_tracing(|| {
            let first = tracing::info_span!("first").in_scope(|| current_traceparent().unwrap());
            let second = tracing::info_span!("second").in_scope(|| current_traceparent().unwrap());
            assert_ne!(trace_id(&first), trace_id(&second));
        });
    }
}
