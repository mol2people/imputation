"""Join derived percentile endpoints to saved HRp10 and run paired inference.

Example:
  python run_inference.py --endpoints run/endpoints/HRp5/expanded90/endpoints.parquet \
    run/endpoints/HRp1/expanded90/endpoints.parquet --output run/inference
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from inference import run_inference
from report import write_report
from secondary import run_secondary

BASELINE=EXP/"artifacts/endpoints/endpoints.parquet"
KEYS=["user_id","source","protocol","window_index"]


def join_saved_reference(alternatives: pd.DataFrame, baseline_path: str|Path=BASELINE) -> pd.DataFrame:
    """Use saved baseline endpoint rows; no HRp10 endpoint replay is performed."""
    baseline=pd.read_parquet(baseline_path).copy()
    baseline["arm"]="HRp10"
    if "selected_dates_hash" not in alternatives or "selected_dates_hash" not in baseline:
        raise ValueError("both alternative and saved reference rows require selected_dates_hash")
    if alternatives.duplicated(KEYS+['arm']).any() or baseline.duplicated(KEYS).any():
        raise ValueError("endpoint rows must be unique by participant/source/protocol/window/arm")
    refcols=KEYS+[c for c in ("age","sex","null_status","selected_dates_hash","n_selected_dates","Q_i","E_i2","A_hat","sigma_resid","Q_i_mc_se","peak_period","start_day") if c in baseline]
    joined=alternatives.merge(baseline[refcols],on=KEYS,how="left",suffixes=("","_reference"),validate="many_to_one")
    if joined.null_status_reference.isna().any():
        raise ValueError("some alternative endpoint keys have no saved HRp10 reference")
    if not joined.selected_dates_hash.eq(joined.selected_dates_hash_reference).all():
        raise ValueError("selected-date hashes differ from saved HRp10 for matched endpoints")
    for col in ("age","sex","start_day"):
        refcol=f"{col}_reference"
        if refcol in joined:
            if not (joined[col].fillna(-999).to_numpy()==joined[refcol].fillna(-999).to_numpy()).all():
                raise ValueError(f"{col} differs from saved HRp10 metadata")
            joined=joined.drop(columns=refcol)
    # The saved HRp10 observations are restricted to the same keys as alternatives.
    keep=alternatives[KEYS].drop_duplicates()
    baseline=baseline.merge(keep,on=KEYS,how="inner",validate="one_to_one")
    return pd.concat([baseline,alternatives],ignore_index=True,sort=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints",nargs="+",required=True,help="alternative-arm endpoints.parquet files")
    parser.add_argument("--baseline",default=str(BASELINE),help="saved HRp10 endpoints.parquet")
    parser.add_argument("--output",required=True)
    parser.add_argument("--run-dir",help="endpoint output root; inferred from <root>/<arm>/<protocol>/endpoints.parquet")
    parser.add_argument("--n-boot",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261020)
    parser.add_argument("--arms",nargs="+",default=["HRp10","HRp5","HRp1"])
    parser.add_argument("--reference-daily",nargs="*",default=[],
                        help="extra daily_values.parquet files (e.g. matched reference) for the agreement table")
    args=parser.parse_args()
    alternatives=pd.concat([pd.read_parquet(p) for p in args.endpoints],ignore_index=True)
    matched=join_saved_reference(alternatives,args.baseline)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    run_inference(matched,output,n_boot=args.n_boot,seed=args.seed,arms=tuple(args.arms))
    run_dir=Path(args.run_dir) if args.run_dir else Path(args.endpoints[0]).resolve().parents[2]
    secondary=run_secondary(matched,run_dir,output,n_boot=args.n_boot,arms=tuple(args.arms))
    daily_frames=[]
    for path in args.endpoints:
        p=Path(path).with_name("daily_values.parquet")
        if p.exists(): daily_frames.append(pd.read_parquet(p))
    for path in args.reference_daily:
        p=Path(path)
        if p.exists(): daily_frames.append(pd.read_parquet(p))
    daily=pd.concat(daily_frames,ignore_index=True) if daily_frames else None
    if daily is not None:
        dkeys=[c for c in ("user_id","source","arm","protocol","window_index","selected_dates_hash","date") if c in daily]
        daily=daily.drop_duplicates(dkeys)
    write_report(output,daily_values=daily,secondary_tables=secondary)
    print(f"Saved paired inference and report under {output}; loaded HRp10 from {args.baseline} without replay.")


if __name__=="__main__":
    main()
