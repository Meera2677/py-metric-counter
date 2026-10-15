"""Core metric primitives with bounded cardinality.

The central design constraint of this module is that every metric object
has a fixed, finite set of label-value combinations and histogram buckets
known at construction time. This means a misbehaving caller passing a
novel label value will raise ``ValueError`` rather than silently growing
the metric set and eventually exhausting memory.

This is a deliberate trade-off: we lose the convenience of "instrument
first, decide the taxonomy later" in exchange for a hard ceiling on the
number of unique time series a single process can emit. In practice the
most common cardinality blow-up is a label that turns out to be
unbounded (user IDs, request paths, full exception strings); requiring
up-front enumeration forces that mistake to surface at startup, not in
production at 3am.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Counter", "Histogram", "BucketingStrategy"]


class Counter:
    """A labelled counter whose label-value set is fixed at construction.

    All allowed (label -> value) combinations are declared when the Counter
    is created. The internal value table is pre-allocated so that
    incrementing never needs to create a new entry and a novel label value
    raises rather than silently expanding the table.

    Values are stored as ``int`` because counter semantics are monotonic
    integer increments; accepting floats would raise questions about NaN,
    rounding, and equality that do not help the typical caller.
    """

    def __init__(
        self,
        name: str,
        label_names: Sequence[str],
        label_value_tuples: Sequence[Tuple[str, ...]],
    ) -> None:
        if not name:
            raise ValueError("Counter name must be a non-empty string")
        if not label_names:
            raise ValueError("Counter requires at least one label name")
        for ln in label_names:
            if not isinstance(ln, str) or not ln:
                raise ValueError("Label names must be non-empty strings")
        for t in label_value_tuples:
            if not isinstance(t, tuple):
                raise TypeError("label_value_tuples must contain tuples")
            if len(t) != len(label_names):
                raise ValueError(
                    "Each label value tuple must match the number of label names"
                )
            for v in t:
                if not isinstance(v, str) or not v:
                    raise ValueError("Label values must be non-empty strings")

        self._name = name
        self._label_names: Tuple[str, ...] = tuple(label_names)
        self._values: Dict[Tuple[str, ...], int] = {
            t: 0 for t in label_value_tuples
        }
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def label_names(self) -> Tuple[str, ...]:
        return self._label_names

    def inc(self, amount: int = 1, **labels: str) -> None:
        """Increment the counter for the given label values.

        Raises ``ValueError`` if the label set is not one of the declared
        combinations, and ``TypeError`` if ``amount`` is not an int.
        """
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError("amount must be an int")
        if amount < 0:
            raise ValueError("amount must be non-negative")
        key = self._key_from_labels(labels)
        with self._lock:
            if key not in self._values:
                raise ValueError(
                    f"Label set {labels} is not declared for counter {self._name!r}"
                )
            self._values[key] += amount

    def value(self, **labels: str) -> int:
        """Return the current value for the given label values."""
        key = self._key_from_labels(labels)
        with self._lock:
            if key not in self._values:
                raise ValueError(
                    f"Label set {labels} is not declared for counter {self._name!r}"
                )
            return self._values[key]

    def snapshot(self) -> Dict[Tuple[str, ...], int]:
        """Return a shallow copy of the current values keyed by label tuples."""
        with self._lock:
            return dict(self._values)

    def _key_from_labels(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        if set(labels.keys()) != set(self._label_names):
            raise ValueError(
                f"Expected labels {set(self._label_names)}, got {set(labels.keys())}"
            )
        return tuple(labels[ln] for ln in self._label_names)


class BucketingStrategy:
    """Pre-declared upper bounds for histogram bucket counts.

    A bucketing strategy is just the sorted list of bucket upper bounds,
    stored as a tuple to make it hashable and immutable. Keeping this as a
    first-class object rather than passing a raw list lets us validate the
    invariants (finite, sorted, strictly increasing) in one place.
    """

    def __init__(self, upper_bounds: Sequence[float]) -> None:
        bounds: List[float] = list(upper_bounds)
        if not bounds:
            raise ValueError("upper_bounds must not be empty")
        for b in bounds:
            if isinstance(b, bool) or not isinstance(b, (int, float)):
                raise TypeError("upper_bounds must be numbers")
            if isinstance(b, float):
                if b != b:
                    raise ValueError("upper_bounds must not contain NaN")
        for i in range(1, len(bounds)):
            if not (bounds[i] > bounds[i - 1]):
                raise ValueError(
                    "upper_bounds must be strictly increasing"
                )
        self._bounds: Tuple[float, ...] = tuple(float(b) for b in bounds)

    @property
    def upper_bounds(self) -> Tuple[float, ...]:
        return self._bounds

    @property
    def n_buckets(self) -> int:
        return len(self._bounds)

    def index_of(self, value: float) -> int:
        """Return the index of the bucket whose upper bound covers ``value``.

        A value equal to an upper bound is assigned to that bucket (the
        upper bound is inclusive). Values above the last bound go to the
        overflow slot, whose index is ``len(upper_bounds)``.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value must be a number")
        if isinstance(value, float) and value != value:
            raise ValueError("value must not be NaN")
        for i, ub in enumerate(self._bounds):
            if value <= ub:
                return i
        return len(self._bounds)


class Histogram:
    """A labelled histogram with fixed buckets and fixed label values.

    The bucket counts, the sum of observations, and the total observation
    count are tracked per label-value tuple. Buckets are inclusive of their
    upper bound and there is an implicit ``+Inf`` overflow bucket which is
    the last element of the count array returned by ``snapshot``.

    The sum is tracked as a float because observations may be fractional;
    callers comparing sums in tests should do so with a tolerance, as the
    README notes.
    """

    def __init__(
        self,
        name: str,
        strategy: BucketingStrategy,
        label_names: Sequence[str],
        label_value_tuples: Sequence[Tuple[str, ...]],
    ) -> None:
        if not name:
            raise ValueError("Histogram name must be a non-empty string")
        if not label_names:
            raise ValueError("Histogram requires at least one label name")
        for ln in label_names:
            if not isinstance(ln, str) or not ln:
                raise ValueError("Label names must be non-empty strings")
        for t in label_value_tuples:
            if not isinstance(t, tuple):
                raise TypeError("label_value_tuples must contain tuples")
            if len(t) != len(label_names):
                raise ValueError(
                    "Each label value tuple must match the number of label names"
                )
            for v in t:
                if not isinstance(v, str) or not v:
                    raise ValueError("Label values must be non-empty strings")

        self._name = name
        self._strategy = strategy
        self._label_names: Tuple[str, ...] = tuple(label_names)
        self._n_slots = strategy.n_buckets + 1  # +1 for overflow
        self._counts: Dict[Tuple[str, ...], Tuple[int, ...]] = {
            t: tuple([0] * self._n_slots) for t in label_value_tuples
        }
        self._sums: Dict[Tuple[str, ...], float] = {t: 0.0 for t in label_value_tuples}
        self._totals: Dict[Tuple[str, ...], int] = {t: 0 for t in label_value_tuples}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def label_names(self) -> Tuple[str, ...]:
        return self._label_names

    @property
    def upper_bounds(self) -> Tuple[float, ...]:
        return self._strategy.upper_bounds

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation under the given label values."""
        key = self._key_from_labels(labels)
        idx = self._strategy.index_of(value)
        fval = float(value)
        with self._lock:
            if key not in self._counts:
                raise ValueError(
                    f"Label set {labels} is not declared for histogram {self._name!r}"
                )
            existing = self._counts[key]
            lst = list(existing)
            lst[idx] += 1
            self._counts[key] = tuple(lst)
            self._sums[key] += fval
            self._totals[key] += 1

    def counts(self, **labels: str) -> Tuple[int, ...]:
        """Return the bucket counts for the given label values.

        The returned tuple has length ``len(upper_bounds) + 1``; the last
        element is the overflow bucket.
        """
        key = self._key_from_labels(labels)
        with self._lock:
            if key not in self._counts:
                raise ValueError(
                    f"Label set {labels} is not declared for histogram {self._name!r}"
                )
            return self._counts[key]

    def sum(self, **labels: str) -> float:
        key = self._key_from_labels(labels)
        with self._lock:
            if key not in self._sums:
                raise ValueError(
                    f"Label set {labels} is not declared for histogram {self._name!r}"
                )
            return self._sums[key]

    def total(self, **labels: str) -> int:
        key = self._key_from_labels(labels)
        with self._lock:
            if key not in self._totals:
                raise ValueError(
                    f"Label set {labels} is not declared for histogram {self._name!r}"
                )
            return self._totals[key]

    def snapshot(self) -> Dict[Tuple[str, ...], Tuple[Tuple[int, ...], float, int]]:
        """Return a copy of all per-label (counts, sum, total) triples."""
        with self._lock:
            return {
                k: (self._counts[k], self._sums[k], self._totals[k])
                for k in self._counts
            }

    def _key_from_labels(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        if set(labels.keys()) != set(self._label_names):
            raise ValueError(
                f"Expected labels {set(self._label_names)}, got {set(labels.keys())}"
            )
        return tuple(labels[ln] for ln in self._label_names)
