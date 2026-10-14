import unittest
import numpy as np
import pandas as pd
from heartbeat_lab.calibration import fit_calibration, probability_m, probability_columns


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame({
            "customer_id": [str(i) for i in range(12)],
            "split": ["val"] * 8 + ["test"] * 4,
            "y": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "score": [-.3, .1, .3, .4, .5, .6, .8, 1.3, -.1, .2, .7, 1.2]})

    def test_test_labels_and_scores_do_not_affect_fit(self):
        before = fit_calibration(self.predictions)
        changed = self.predictions.copy()
        changed.loc[changed.split.eq("test"), "y"] = 1 - changed.loc[changed.split.eq("test"), "y"]
        changed.loc[changed.split.eq("test"), "score"] = 999
        self.assertEqual(before, fit_calibration(changed))
        with self.assertRaises(ValueError):
            fit_calibration(pd.concat([self.predictions, self.predictions.iloc[:1]]))

    def test_extremes_missing_values_and_raw_scores(self):
        calibration = fit_calibration(self.predictions)
        values = np.array([-1e6, -.3, .5, 1.3, 1e6, np.nan])
        p = probability_m(values, calibration)
        self.assertTrue(np.isnan(p[-1]))
        self.assertTrue(((p[:-1] >= 0) & (p[:-1] <= 1)).all())
        self.assertTrue((np.diff(p[:-1]) >= 0).all())
        source = pd.DataFrame({"model": "example", "daily_score": values, "cumulative_score": values})
        result = probability_columns(source, {"example": calibration})
        pd.testing.assert_frame_equal(result[source.columns], source)
        np.testing.assert_equal(result.daily_p_m, p)


if __name__ == "__main__":
    unittest.main()
