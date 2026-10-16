# py_metric_counter

A small library for counters and histograms with bounded cardinality: all allowed label-value combinations and histogram buckets must be declared up front, so a novel label value raises rather than silently expanding the metric set.

## Usage

```python
from py_metric_counter import Counter, Histogram, BucketingStrategy

c = Counter(
    "http_requests_total",
    ("method", "status"),
    [("GET", "200"), ("GET", "500"), ("POST", "200")],
)
c.inc(method="GET", status="200")
c.inc(3, method="POST", status="200")
assert c.value(method="GET", status="200") == 1
assert c.value(method="POST", status="200") == 3

h = Histogram(
    "request_latency_seconds",
    BucketingStrategy([0.1, 0.5, 1.0, 5.0]),
    ("route",),
    [("/a",), ("/b",)],
)
h.observe(0.05, route="/a")
h.observe(0.75, route="/a")
h.observe(7.0, route="/a")
counts = h.counts(route="/a")  # (1, 1, 0, 0, 1) — last entry is the +Inf overflow bucket
```

## Why this exists

Cardinality blow-up is the most common way a metrics-instrumented service quietly degrades in production: a label that looked bounded ("route", "error code") turns out to be a proxy for something unbounded (full query strings, per-user identifiers). By the time the alert fires, the in-memory metric map has already grown past the point where the process can recover gracefully.

This library makes the trade-off explicit: every label-value combination must be declared at construction, and histogram buckets must be a strictly increasing, finite, non-NaN sequence. The cost is verbosity at instrumentation sites and a startup-time failure if the taxonomy is wrong. The benefit is a hard ceiling on the number of unique time series a single metric can produce — there is no code path that can add a new one at runtime.

Counter increments are integer-only; this avoids questions about NaN and float equality in a context where the only sensible operation is monotonic addition. Histogram sums are floats because fractional observations are legitimate, but tests comparing sums should use a tolerance rather than `==`.

## Edge cases you will hit

- A counter increment with a label value not in the declared set raises `ValueError` at observation time, not silently at startup. Declare the combination first.
- A histogram value exactly equal to a bucket upper bound falls into that bucket (upper bounds are inclusive). Values above the last bound go into an implicit `+Inf` overflow bucket, which is the final element of the tuple returned by `counts`.
- `BucketingStrategy` rejects empty, non-increasing, or NaN-containing bound lists at construction; `Histogram.observe` likewise rejects NaN observations. There is no silent fallback to an overflow bucket for NaN.
- `bool` is explicitly rejected as a counter increment amount, despite `isinstance(True, int)` being `True` in Python — the intent of a boolean is not a numeric increment.
