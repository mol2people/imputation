from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from cohort_allocator import (
    allocate_cohort,
    fold_target_counts,
    make_config,
    write_prepared_cohorts,
)
from cohort_allocator.cli import main


def _demographics() -> pd.DataFrame:
    rows = []
    for source, size in (("Apple", 31), ("Garmin", 27)):
        for number in range(size):
            rows.append(
                {
                    "participant_id": f"{source}-{number:03d}",
                    "source": source,
                    "salutation": 10 if number % 3 else 20,
                    "age_band_5y": f"{25 + 5 * (number % 4)}-{29 + 5 * (number % 4)}",
                    "bmi_group": ["low", "mid", "high", None][number % 4],
                    "unrelated_note": f"kept-{number}",
                }
            )
    return pd.DataFrame(rows)


def _config():
    return make_config(
        id_column="participant_id",
        stratify_columns=("salutation", "age_band_5y", "bmi_group"),
        fold_sizes=(("train", 8), ("validation", 1), ("test", 1)),
        partition_columns=("source",),
        seed=887,
        time_limit_seconds=15,
    )


class AllocatorTests(unittest.TestCase):
    def test_partitioned_three_fold_allocation_is_exact_and_reproducible(self) -> None:
        demographics = _demographics()
        first = allocate_cohort(demographics.sample(frac=1, random_state=2), _config())
        second = allocate_cohort(demographics.sample(frac=1, random_state=13), _config())

        first_assignments = first.assignments.sort_values("participant_id").reset_index(drop=True)
        second_assignments = second.assignments.sort_values("participant_id").reset_index(drop=True)
        pdt.assert_frame_equal(first_assignments, second_assignments)
        self.assertTrue(first.assignments["participant_id"].is_unique)
        self.assertEqual(set(first.assignments["fold"]), {"train", "validation", "test"})
        self.assertTrue(first.assignments["unrelated_note"].notna().all())

        for source, group in first.assignments.groupby("source"):
            expected = fold_target_counts(len(group), (0.8, 0.1, 0.1))
            observed = [int((group["fold"] == fold).sum()) for fold in ("train", "validation", "test")]
            self.assertEqual(observed, expected, source)

        self.assertEqual(
            {"joint", "salutation", "age_band_5y", "bmi_group"},
            set(first.balance["balance_family"]),
        )
        self.assertTrue(all(partition["all_phases_optimal"] for partition in first.summary["partitions"]))

    def test_cli_writes_prepared_csv_cohorts_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "demographics.csv"
            _demographics().to_csv(summary, index=False)
            output = root / "prepared"
            exit_code = main(
                [
                    str(summary),
                    "--id-column",
                    "participant_id",
                    "--stratify",
                    "salutation,age_band_5y,bmi_group",
                    "--partition-by",
                    "source",
                    "--fold-sizes",
                    "train=3,test=1",
                    "--seed",
                    "887",
                    "--time-limit",
                    "15",
                    "--format",
                    "csv",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "assignments.csv").is_file())
            self.assertTrue((output / "balance.csv").is_file())
            self.assertTrue((output / "cohorts" / "train.csv").is_file())
            self.assertTrue((output / "cohorts" / "test.csv").is_file())
            payload = json.loads((output / "allocation_summary.json").read_text())
            self.assertEqual(payload["n_participants"], 58)
            self.assertEqual(payload["config"]["fold_sizes"], {"train": 0.75, "test": 0.25})

    def test_nonempty_output_directory_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protected"
            output.mkdir()
            (output / "unrelated.txt").write_text("keep me")
            result = allocate_cohort(_demographics(), _config())
            with self.assertRaisesRegex(ValueError, "not empty"):
                write_prepared_cohorts(result, output, output_format="csv")
