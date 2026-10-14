"""Prepare one small NPZ per customer from a labelled CSV manifest.

Manifest: customer_id,salutation,csv_path. Paths are relative to the manifest.
Only eligible customers (>=2 adequate days) enter allocation. Inspect exclusions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from heartbeat_lab import prepare_days
from heartbeat_lab.data import STATS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"customer_id": str})
    if not {"customer_id", "salutation", "csv_path"}.issubset(manifest):
        raise ValueError("Need customer_id,salutation,csv_path.")
    if manifest.customer_id.isna().any() or manifest.customer_id.duplicated().any():
        raise ValueError("Customer IDs must be nonmissing and unique.")
    if not manifest.salutation.isin([10, 20]).all():
        raise ValueError("Salutation must be 10 or 20.")
    args.output.mkdir(parents=True, exist_ok=False)
    included, excluded = [], []
    for index, row in enumerate(manifest.itertuples(index=False)):
        try:
            frame = pd.read_csv(args.manifest.parent / row.csv_path)
            days, hr, quality = prepare_days(frame)
            if len(days) < 2:
                raise ValueError("Need at least two adequate days for training.")
        except (ValueError, OSError) as error:
            excluded.append({"customer_id": row.customer_id, "reason": str(error)})
            continue
        filename = f"customer_{index:06d}.npz"  # never use IDs as filesystem paths
        np.savez_compressed(args.output / filename, hr=hr,
                            date=days.date.to_numpy(dtype="datetime64[D]").astype("int64"),
                            stats=days[STATS].to_numpy(np.float32))
        included.append(dict(customer_id=row.customer_id, y=int(row.salutation == 20),
                              file=filename, **quality))
    pd.DataFrame(included, columns=["customer_id", "y", "file", "input_rows", "retained_rows", "rejected_rows", "observed_days", "adequate_days"]).to_csv(args.output / "customers.csv", index=False)
    pd.DataFrame(excluded, columns=["customer_id", "reason"]).to_csv(args.output / "exclusions.csv", index=False)
    print(f"Prepared {len(included)} customers; excluded {len(excluded)}. Review exclusions.csv.")


if __name__ == "__main__":
    main()
