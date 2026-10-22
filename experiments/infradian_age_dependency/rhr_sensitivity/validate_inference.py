"""Bounded synthetic validation of paired RHR inference (not cohort analysis)."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from inference import run_inference


def main() -> None:
    rng=np.random.default_rng(20261020); rows=[]
    for band in range(10):
        for sex in (0,1):
            for j in range(30):
                uid=band*60+sex*30+j
                age=25+5*band+0.1+j*0.01
                signal=0.01+0.001*sex*np.sin(age/7)+rng.normal(0,0.002)
                for protocol in ("expanded90","balanced90"):
                    for arm,shift in (("HRp10",0.),("HRp5",0.),("HRp1",0.25)):
                        rows.append({"user_id":uid,"source":6,"arm":arm,"protocol":protocol,
                            "window_index":1,"age":age,"sex":sex,"null_status":"converged",
                            "selected_dates_hash":f"{protocol}-fixed","n_selected_dates":90,
                            "Q_i":signal+shift,"E_i2":2*signal+shift,"A_hat":3*signal+shift,
                            "sigma_resid":4*signal+shift})
    output=Path(tempfile.mkdtemp(prefix="rhr-inference-smoke-"))
    tables=run_inference(pd.DataFrame(rows),output,n_boot=50)
    assert len(tables["cohort_counts"])==2
    assert tables["cohort_counts"].status.eq("ok").all()
    for arm,expected in (("HRp5",0.),("HRp1",0.25)):
        female=tables["paired_curve_differences"].query(
            "arm == @arm and endpoint == 'Q_i' and curve == 'female'")
        delta=tables["paired_curve_differences"].query(
            "arm == @arm and endpoint == 'Q_i' and curve == 'delta'")
        assert np.allclose(female.difference,expected,atol=1e-7)
        assert np.allclose(delta.difference,0.,atol=1e-7)
    tests=tables["chunk_tests"]
    assert len(tests)==3*4*4*2
    assert tests[["f_hc3","p_f","p_bootstrap_centered"]].notna().all().all()
    assert len(tables["paired_transitions"])==2*2
    print("PASS: common support, shared bootstrap, zero and known-shift paired curves, paired transitions, four tests with HC3 F and centered-bootstrap p-values")
    print(f"Temporary outputs: {output}")


if __name__=="__main__":
    main()
