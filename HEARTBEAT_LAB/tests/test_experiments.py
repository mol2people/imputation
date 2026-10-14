import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from heartbeat_lab.data import normalize_epochs, prepare_days
from heartbeat_lab.model import HeartbeatModel
from heartbeat_lab.perturb import perturb_events, repeated_predictions, shuffle_bins
from heartbeat_lab.splits import temporal_customers


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        raw = pd.read_csv(Path(__file__).resolve().parents[1] / "examples/synthetic_hr.csv")
        self.events, _ = normalize_epochs(raw)
        self.days, self.hr, _ = prepare_days(raw)

    def test_shuffle_preserves_values_mask_and_unselected_events(self):
        selected = [self.days.date.iloc[0]]
        result = perturb_events(self.events, selected, "shuffle", 42)
        chosen = self.events.date.isin(selected)
        pd.testing.assert_frame_equal(result.loc[~chosen], self.events.loc[~chosen])
        np.testing.assert_array_equal(np.sort(result.loc[chosen].longValue), np.sort(self.events.loc[chosen].longValue))
        pd.testing.assert_frame_equal(result.drop(columns="longValue"), self.events.drop(columns="longValue"))
        pd.testing.assert_frame_equal(result, perturb_events(self.events, selected, "shuffle", 42))
        shuffled = shuffle_bins(self.hr, 42)
        np.testing.assert_array_equal(np.isfinite(shuffled), np.isfinite(self.hr))
        np.testing.assert_array_equal(np.sort(shuffled, axis=1), np.sort(self.hr, axis=1))

    def test_resample_shift_and_thinning(self):
        selected = self.days.date.tolist()
        sampled = perturb_events(self.events, selected, "resample", 42)
        self.assertEqual(len(sampled), len(self.events))
        for date, day in sampled.groupby("date"):
            self.assertTrue(set(day.longValue) <= set(self.events.loc[self.events.date.eq(date), "longValue"]))
        shifted = perturb_events(self.events, selected, "shift", 42)
        np.testing.assert_array_equal(shifted.endTimestamp-shifted.startTimestamp,
                                      self.events.endTimestamp-self.events.startTimestamp)
        local = shifted.startTimestamp + shifted.timezoneOffset*60_000
        np.testing.assert_array_equal(pd.to_datetime(local, unit="ms").dt.normalize(), shifted.date)
        self.assertTrue(perturb_events(self.events, selected, "thin", 42, fraction=1).empty)

    def test_repeated_predictions_and_dropped_days(self):
        customers = []
        for i in range(8):
            d = self.days.copy()
            d[["mean", "median", "vmin", "vmax"]] += i
            customers.append((d, self.hr+i))
        models = {"summary": HeartbeatModel(representation="summary").fit(customers, [0, 1]*4)}
        base, runs = repeated_predictions(self.events, self.days, self.hr, models,
                                          self.days.date.tolist(), "shuffle", repeats=2)
        for _, group in runs.groupby("run"):
            np.testing.assert_allclose(group.daily_score, base.daily_score, atol=1e-6)
        _, dropped = repeated_predictions(self.events, self.days, self.hr, models,
                                           self.days.date.tolist(), "thin", repeats=2, fraction=1)
        self.assertTrue(dropped.daily_score.isna().all())
        self.assertFalse(dropped.adequate.any())
        _, partial = repeated_predictions(self.events, self.days, self.hr, models,
                                           [self.days.date.iloc[-1]], "thin", repeats=2, fraction=1)
        self.assertTrue(partial.loc[partial.selected_day, "daily_score"].isna().all())
        self.assertTrue(partial.loc[~partial.selected_day, "daily_score"].notna().all())

    def test_temporal_customers_cut_histories_and_keep_ids_disjoint(self):
        customers = pd.DataFrame({"customer_id": [str(i) for i in range(6)], "y": [0, 1]*3})
        histories = []
        for month in [1, 1, 3, 3, 5, 5]:
            dates = pd.to_datetime([f"2020-{month:02d}-01", f"2020-{month:02d}-02", "2020-12-31"])
            histories.append((pd.DataFrame({"date": dates}), np.ones((3, 288))))
        assigned, data, excluded = temporal_customers(customers, histories, "2020-03-01", "2020-05-01")
        self.assertEqual(assigned.split.tolist(), ["train", "train", "val", "val", "test", "test"])
        self.assertTrue(excluded.empty)
        self.assertLess(max(d.date.max() for d, _ in data[:2]), pd.Timestamp("2020-03-01"))
        self.assertLess(max(d.date.max() for d, _ in data[2:4]), pd.Timestamp("2020-05-01"))
        self.assertEqual([len(d) for d, _ in data], [2, 2, 2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
