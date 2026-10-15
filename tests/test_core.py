import math
import threading
import unittest

from py_metric_counter import Counter, Histogram, BucketingStrategy


class TestCounter(unittest.TestCase):
    def test_increment_default(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        c.inc(method="GET")
        self.assertEqual(c.value(method="GET"), 1)
        self.assertEqual(c.value(method="POST"), 0)

    def test_increment_custom_amount(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        c.inc(5, method="POST")
        self.assertEqual(c.value(method="POST"), 5)

    def test_increment_repeated(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        c.inc(2, method="GET")
        c.inc(3, method="GET")
        self.assertEqual(c.value(method="GET"), 5)

    def test_unknown_label_value_raises(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        with self.assertRaises(ValueError):
            c.inc(method="PUT")

    def test_missing_label_raises(self):
        c = Counter("requests", ("method", "status"), [("GET", "200"), ("POST", "500")])
        with self.assertRaises(ValueError):
            c.inc(method="GET")

    def test_extra_label_raises(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        with self.assertRaises(ValueError):
            c.inc(method="GET", status="200")

    def test_negative_amount_raises(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        with self.assertRaises(ValueError):
            c.inc(-1, method="GET")

    def test_non_int_amount_raises(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        with self.assertRaises(TypeError):
            c.inc(1.5, method="GET")

    def test_bool_amount_rejected(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        with self.assertRaises(TypeError):
            c.inc(True, method="GET")

    def test_snapshot_is_copy(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        c.inc(method="GET")
        snap = c.snapshot()
        snap[("GET",)] = 999
        self.assertEqual(c.value(method="GET"), 1)

    def test_concurrent_increments(self):
        c = Counter("requests", ("method",), [("GET",), ("POST",)])
        def worker():
            for _ in range(1000):
                c.inc(method="GET")
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(c.value(method="GET"), 8000)


class TestBucketingStrategy(unittest.TestCase):
    def test_basic_bounds(self):
        s = BucketingStrategy([0.1, 0.5, 1.0, 5.0])
        self.assertEqual(s.upper_bounds, (0.1, 0.5, 1.0, 5.0))
        self.assertEqual(s.n_buckets, 4)

    def test_index_below_first(self):
        s = BucketingStrategy([1.0, 2.0])
        self.assertEqual(s.index_of(0.5), 0)

    def test_index_at_bound_is_inclusive(self):
        s = BucketingStrategy([1.0, 2.0])
        self.assertEqual(s.index_of(1.0), 0)
        self.assertEqual(s.index_of(2.0), 1)

    def test_index_overflow(self):
        s = BucketingStrategy([1.0, 2.0])
        self.assertEqual(s.index_of(2.5), 2)

    def test_empty_bounds_rejected(self):
        with self.assertRaises(ValueError):
            BucketingStrategy([])

    def test_non_increasing_rejected(self):
        with self.assertRaises(ValueError):
            BucketingStrategy([1.0, 1.0])
        with self.assertRaises(ValueError):
            BucketingStrategy([2.0, 1.0])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            BucketingStrategy([float("nan")])
        s = BucketingStrategy([1.0, 2.0])
        with self.assertRaises(ValueError):
            s.index_of(float("nan"))

    def test_integer_input_accepted(self):
        s = BucketingStrategy([1, 2, 3])
        self.assertEqual(s.index_of(2), 1)

    def test_non_number_rejected(self):
        s = BucketingStrategy([1.0, 2.0])
        with self.assertRaises(TypeError):
            s.index_of("not a number")


class TestHistogram(unittest.TestCase):
    def test_observe_and_counts(self):
        h = Histogram(
            "latency",
            BucketingStrategy([1.0, 2.0]),
            ("route",),
            [("/a",), ("/b",)],
        )
        h.observe(0.5, route="/a")
        h.observe(1.5, route="/a")
        h.observe(2.5, route="/a")
        counts = h.counts(route="/a")
        self.assertEqual(counts, (1, 1, 1))
        self.assertEqual(h.total(route="/a"), 3)

    def test_bound_is_inclusive(self):
        h = Histogram("latency", BucketingStrategy([1.0]), ("route",), [("/a",)])
        h.observe(1.0, route="/a")
        self.assertEqual(h.counts(route="/a"), (1, 0))

    def test_overflow_bucket(self):
        h = Histogram("latency", BucketingStrategy([1.0, 2.0]), ("route",), [("/a",)])
        h.observe(3.0, route="/a")
        counts = h.counts(route="/a")
        self.assertEqual(counts, (0, 0, 1))

    def test_sum_and_total(self):
        h = Histogram("latency", BucketingStrategy([1.0, 2.0]), ("route",), [("/a",)])
        h.observe(0.5, route="/a")
        h.observe(1.5, route="/a")
        self.assertAlmostEqual(h.sum(route="/a"), 2.0)
        self.assertEqual(h.total(route="/a"), 2)

    def test_unknown_label_value_raises(self):
        h = Histogram("latency", BucketingStrategy([1.0]), ("route",), [("/a",)])
        with self.assertRaises(ValueError):
            h.observe(1.0, route="/x")

    def test_counts_unknown_label_raises(self):
        h = Histogram("latency", BucketingStrategy([1.0]), ("route",), [("/a",)])
        with self.assertRaises(ValueError):
            h.counts(route="/x")

    def test_multiple_label_sets_isolated(self):
        h = Histogram(
            "latency",
            BucketingStrategy([1.0]),
            ("route", "status"),
            [("/a", "200"), ("/a", "500")],
        )
        h.observe(0.5, route="/a", status="200")
        self.assertEqual(h.counts(route="/a", status="200"), (1, 0))
        self.assertEqual(h.counts(route="/a", status="500"), (0, 0))

    def test_snapshot_keys(self):
        h = Histogram(
            "latency",
            BucketingStrategy([1.0]),
            ("route",),
            [("/a",), ("/b",)],
        )
        h.observe(0.5, route="/a")
        snap = h.snapshot()
        self.assertIn(("/a",), snap)
        self.assertIn(("/b",), snap)
        counts, s, t = snap[("/a",)]
        self.assertEqual(counts, (1, 0))
        self.assertAlmostEqual(s, 0.5)
        self.assertEqual(t, 1)


if __name__ == "__main__":
    unittest.main()
