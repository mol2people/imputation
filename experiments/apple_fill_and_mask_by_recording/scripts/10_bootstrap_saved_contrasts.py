"""Conditional exploratory paired bootstrap of saved scores; no model fitting.

1000 class-stratified participant resamples per subgroup, seed 20261019.
Percentile intervals exclude fitting/selection uncertainty and multiplicity correction.
Membership uses script09's verified full-history 5m descriptors and saved cuts.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT=Path(__file__).resolve().parents[1]/'results'

def main():
    p=pd.read_parquet(OUT/'test_predictions.parquet').query("penalty_type == 'ridge_selected_ext'")
    p['user_id']=p.user_id.astype(str)
    info=pd.read_csv(OUT/'reconciled_test_recording_descriptors.csv',dtype={'user_id':str}).set_index('user_id')
    w=p.pivot(index='user_id',columns=['budget','resolution','arm'],values='score')
    y=p.drop_duplicates('user_id').set_index('user_id').y.loc[w.index]
    assert not w.isna().any().any()
    rows=[]
    groups=[('overall','all',np.ones(len(w),bool))]+[('occupancy',level,(info.loc[w.index,'occupancy']==level).to_numpy()) for level in ['low','middle','high']]
    for dimension,subgroup,mask in groups:
        yy=y.to_numpy()[mask];ww=w.loc[mask];rng=np.random.default_rng(20261019)
        pairs={}
        for res in ['5m','15m']:
            for arm in ['H_global','H_clean_diff','H_short_interp','H_plus_M_concat','H_plus_M_stack']:
                pairs[res+' '+arm+'_minus_H_clock']=(('all_days',res,arm),('all_days',res,'H_clock'))
        if dimension=='overall':
            for res in ['5m','15m']:pairs[res+' M_temporal_minus_M_static']=(('all_days',res,'M_temporal'),('all_days',res,'M_static'))
            for arm in ['H_clock','H_plus_M_stack']:
                pairs['5m_minus_15m '+arm]=(('all_days','5m',arm),('all_days','15m',arm))
                pairs['all_days_minus_40d '+arm]=(('all_days','5m',arm),('up_to_40d','5m',arm))
        needed=set(c for pair in pairs.values() for c in pair)
        arrays={c:ww[c].to_numpy() for c in needed};dist={name:[] for name in pairs}
        ix0=np.flatnonzero(yy==0);ix1=np.flatnonzero(yy==1)
        for _ in range(1000):
            ix=np.r_[rng.choice(ix0,len(ix0),True),rng.choice(ix1,len(ix1),True)]
            auc={c:roc_auc_score(yy[ix],score[ix]) for c,score in arrays.items()}
            for name,(lhs,rhs) in pairs.items():dist[name].append(auc[lhs]-auc[rhs])
        for name,(lhs,rhs) in pairs.items():
            lo,hi=np.quantile(dist[name],[.025,.975])
            rows.append({'allocation':0,'penalty_type':'ridge_selected_ext','dimension':dimension,'subgroup':subgroup,'contrast':name,'n_test':len(yy),'n_class_0':len(ix0),'n_class_1':len(ix1),'delta_auc':roc_auc_score(yy,arrays[lhs])-roc_auc_score(yy,arrays[rhs]),'ci_lower':lo,'ci_upper':hi,'n_bootstrap':1000,'seed':20261019,'interval':'conditional stratified paired percentile 95%; exploratory; no multiplicity adjustment'})
    pd.DataFrame(rows).to_csv(OUT/'conditional_bootstrap_contrasts.csv',index=False)
    sources=[OUT/'test_predictions.parquet',OUT/'reconciled_test_recording_descriptors.csv']
    provenance={'script':str(Path(__file__).name),'output':'conditional_bootstrap_contrasts.csv','allocations':[0],'n_bootstrap':1000,'seed':20261019,'method':'Within each group, resample each class with replacement at its observed fixed count. Use identical resampled participant indices across every compared score. Reset RNG seed per group; all contrasts within group share draws. Percentile .025/.975 intervals.','scope':'Exploratory conditional saved-predictor AUROC differences; no fitting, selection, subgroup-discovery uncertainty or multiplicity adjustment.','input_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}}
    (OUT/'BOOTSTRAP_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(rows).query("dimension=='occupancy' and contrast=='5m H_global_minus_H_clock'").to_string(index=False))

if __name__=='__main__':main()
