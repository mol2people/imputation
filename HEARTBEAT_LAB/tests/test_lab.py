"""Small behavioural checks: calendar/bin semantics and leakage boundaries."""
import unittest
import numpy as np
import pandas as pd
from heartbeat_lab import allocate_customers, customer_cv, prepare_days, validate_assignments
from heartbeat_lab.model import HeartbeatModel


def example(days=3, offset=0):
    time = pd.date_range("2026-09-01", periods=days*288, freq="5min", tz="Europe/Berlin")
    frame = pd.DataFrame({"timestamp": time.astype(str),
                          "hr": 70 + offset + 8*np.sin(np.arange(len(time))/40)})
    return prepare_days(frame)[:2]


class LabTests(unittest.TestCase):
    def test_csv_native_equivalence_and_timezone(self):
        time = pd.date_range("2026-09-01", periods=288, freq="5min", tz="Europe/Berlin")
        simple = pd.DataFrame({"timestamp": time.astype(str), "hr": np.arange(288) % 20 + 60})
        native = pd.DataFrame(dict(startTimestamp=time.as_unit("ms").asi8,
                                   endTimestamp=time.as_unit("ms").asi8,
                                   timezoneOffset=120, longValue=simple.hr, source=3, type=3000))
        a, ah, _ = prepare_days(simple)
        b, bh, _ = prepare_days(native)
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(ah, bh)
        self.assertEqual(a.date.iloc[0], pd.Timestamp("2026-09-01"))
        with self.assertRaisesRegex(ValueError, "offset"):
            prepare_days(simple.assign(timestamp="2026-09-01T00:00:00"))

    def test_adequacy_missingness_and_filtering(self):
        time = pd.date_range("2026-09-01", periods=96, freq="10min", tz="UTC")
        frame = pd.DataFrame({"timestamp": time.astype(str), "hr": 70})
        _, hr, quality = prepare_days(frame)
        self.assertEqual(np.isfinite(hr).sum(), 96)
        self.assertEqual(quality["adequate_days"], 1)
        with self.assertRaisesRegex(ValueError, "adequate"):
            prepare_days(frame.head(47))
        with self.assertRaisesRegex(ValueError, "valid"):
            prepare_days(frame.assign(hr=999))

    def test_customer_splits_and_cv(self):
        rows = pd.DataFrame({"customer_id": np.repeat(np.arange(80), 3), "y": np.repeat(np.arange(80) % 2, 3)})
        assignments = allocate_customers(rows)
        pd.testing.assert_frame_equal(assignments, allocate_customers(rows.sample(frac=1, random_state=5)))
        self.assertEqual(len(assignments), 80)
        self.assertEqual(assignments.split.value_counts().to_dict(), {"train": 56, "val": 12, "test": 12})
        train = rows[rows.customer_id.isin(assignments.loc[assignments.split.eq("train"), "customer_id"])]
        seen = []
        for tr, va in customer_cv(train, 4):
            self.assertFalse(set(train.iloc[tr].customer_id) & set(train.iloc[va].customer_id))
            seen.extend(va.tolist())
        self.assertEqual(sorted(seen), list(range(len(train))))
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_assignments(rows, pd.concat([assignments, assignments.head(1)]))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            allocate_customers(pd.concat([rows, pd.DataFrame({"customer_id": [0], "y": [1]})]))

    def test_summary_timeline_no_future_data_and_serialization(self):
        import joblib
        import tempfile
        from pathlib import Path
        customers = [example(offset=i) for i in range(8)]
        model = HeartbeatModel(representation="summary").fit(customers, [0, 1]*4)
        days, hr = example(4)
        timeline = model.timeline(days, hr)
        prefix = model.timeline(days.iloc[:2], hr[:2])
        np.testing.assert_allclose(timeline.cumulative_score.iloc[:2], prefix.cumulative_score)
        for i in range(len(days)):
            self.assertAlmostEqual(timeline.cumulative_score.iloc[i],
                                   model.predict([(days.iloc[:i+1], hr[:i+1])])[0], places=6)
            self.assertAlmostEqual(timeline.daily_score.iloc[i],
                                   model.predict([(days.iloc[i:i+1], hr[i:i+1])])[0], places=6)
        self.assertAlmostEqual(timeline.cumulative_score.iloc[-1], model.predict([(days, hr)])[0], places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(joblib.load(path).predict(customers), model.predict(customers))


if __name__ == "__main__":
    unittest.main()
