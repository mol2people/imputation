"""Reconcile ranking summaries from saved allocation-0 predictions; never fit models.

Uses saved training-defined cuts and full-history 5m masks for subgroup membership.
Cross-allocation SDs are undefined for one saved allocation and remain missing.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / 'results'
ART = ROOT / 'experiments/apple_cohort_scaling/artifacts'
KEY = ['allocation', 'resolution', 'budget', 'arm', 'penalty_type']
AXES = {'history':'n_days', 'occupancy':'occupancy_rate_5m',
        'continuity':'adequate_day_density', 'occupancy_variability':'daily_occupancy_sd'}


def main():
    pred_path = OUT / 'test_predictions.parquet'
    p = pd.read_parquet(pred_path)
    assert set(p.allocation) == {0}, 'Review aggregation before adding allocations.'
    p['user_id'] = p.user_id.astype(str)
    assert not p.duplicated(KEY + ['user_id']).any()
    assert p.groupby('user_id').y.nunique().max() == 1
    assert np.isfinite(p.score).all()
    customers = pd.read_csv(ART / 'cache_5min/customers.csv', dtype={'customer_id':str}).set_index('customer_id')
    allocations = pd.read_parquet(ART / 'apple_10_allocations.parquet')
    allocations['customer_id'] = allocations.customer_id.astype(str)
    allocations = allocations.set_index('customer_id')
    ids = pd.Index(sorted(p.user_id.unique()))
    assert set(ids) == set(allocations.index[allocations.alloc_0 == 'test'])
    info = customers.loc[ids].copy()
    for uid in ids:
        with np.load(ART / 'cache_5min' / info.loc[uid, 'file']) as d:
            dates = d['date']; occupancy = np.isfinite(d['hr']).mean(axis=1)
        assert len(dates) == info.loc[uid, 'n_days']
        assert np.isclose(occupancy.mean(), info.loc[uid, 'occupancy_rate_5m'], rtol=0, atol=1e-14)
        info.loc[uid, 'adequate_day_density'] = len(dates) / (int(dates[-1] - dates[0]) + 1)
        info.loc[uid, 'daily_occupancy_sd'] = occupancy.std(ddof=1) if len(dates)>1 else 0
    cuts_path = ART / 'binning_by_recording/training_cutpoints.csv'
    cuts = pd.read_csv(cuts_path).query('allocation == 0').set_index('dimension')
    train_ids = allocations.index[allocations.alloc_0 == 'train']
    for axis in ['history', 'occupancy']:
        actual = customers.loc[train_ids, AXES[axis]].quantile([1/3, 2/3]).to_numpy()
        assert np.allclose(actual, cuts.loc[axis, ['lower','upper']].to_numpy(float), rtol=0, atol=1e-14)
    labels = {}
    for axis, column in AXES.items():
        lo, hi = cuts.loc[axis, ['lower','upper']]
        labels[axis] = np.where(info[column]<=lo, 'low', np.where(info[column]<=hi, 'middle','high'))
    groups = [('overall','all',np.ones(len(ids),bool))]
    for axis in AXES:
        groups.extend((axis,level,labels[axis]==level) for level in ['low','middle','high'])
    for axis in ['history','continuity']:
        groups.extend((axis+'_x_occupancy',a+'/'+b,(labels[axis]==a)&(labels['occupancy']==b))
                      for a in ['low','middle','high'] for b in ['low','middle','high'])
    rows=[]
    metrics=pd.read_csv(OUT/'benchmark_metrics.csv').set_index(KEY)
    for config, g in p.groupby(KEY):
        g=g.set_index('user_id').loc[ids]
        assert len(g)==1671 and not g.score.isna().any()
        auc=roc_auc_score(g.y,g.score); ap=average_precision_score(g.y,g.score)
        assert abs(auc-metrics.loc[config,'test_auc'])<1e-14
        assert abs(ap-metrics.loc[config,'test_ap'])<1e-14
        for dimension, subgroup, mask in groups:
            y=g.y.to_numpy()[mask]; score=g.score.to_numpy()[mask]
            rows.append(dict(zip(KEY,config)) | {'dimension':dimension,'subgroup':subgroup,
                'n_test':len(y),'n_class_0':int((y==0).sum()),'n_class_1':int((y==1).sum()),
                'auroc':roc_auc_score(y,score) if len(np.unique(y))==2 else np.nan,
                'ap':average_precision_score(y,score) if len(np.unique(y))==2 else np.nan})
    direct=pd.DataFrame(rows)
    assert len(direct)==2976
    summary=direct.drop(columns=['allocation']).rename(columns={'n_test':'n_test_mean','n_class_0':'n_class_0_mean','n_class_1':'n_class_1_mean','auroc':'auroc_mean','ap':'ap_mean'})
    summary['n_allocations']=1
    for col in ['auroc_sd','ap_sd','auroc_alloc_1','auroc_alloc_2']:summary[col]=np.nan
    for col in ['auroc_min','auroc_max','auroc_alloc_0']:summary[col]=summary.auroc_mean
    old_columns=pd.read_csv(OUT/'subgroup_summary.csv',nrows=0).columns
    summary=summary[list(old_columns)]
    contrast=[]
    pairs=[('clock_vs_global','H_global','H_clock'),('withholding_gap_diff','H_clean_diff','H_clock'),('withholding_gap_diff','H_clean_diff','H_global'),('short_gap_interp','H_short_interp','H_clock'),('mask_rep_increment','M_temporal','M_static'),('mask_vs_mixed_baseline','M_static','H_clock'),('mask_vs_mixed_baseline','M_temporal','H_clock'),('joint_concat_and_stack','H_plus_M_concat','H_clock'),('joint_concat_and_stack','H_plus_M_stack','H_clock'),('joint_concat_and_stack','H_plus_M_stack','H_plus_M_concat')]
    def add(budget,penalty,res,dim,sub,category,name,lhs,rhs,delta):
        contrast.append({'budget':budget,'penalty_type':penalty,'resolution':res,'contrast_category':category,'contrast_name':name,'comparison_arm':lhs,'baseline_arm':rhs,'dimension':dim,'subgroup':sub,'n_allocations':1,'delta_mean':delta,'delta_sd':np.nan,'delta_min':delta,'delta_max':delta,'sign_agreement':'1/1 '+('+' if delta>1e-12 else '-' if delta < -1e-12 else '0'),'delta_alloc_0':delta,'delta_alloc_1':np.nan,'delta_alloc_2':np.nan})
    for (budget,penalty,res,dim,sub),g in direct.groupby(['budget','penalty_type','resolution','dimension','subgroup']):
        values=g.set_index('arm').auroc
        for category,lhs,rhs in pairs:add(budget,penalty,res,dim,sub,category,lhs+'_minus_'+rhs,lhs,rhs,values[lhs]-values[rhs])
    for (budget,penalty,arm,dim,sub),g in direct.groupby(['budget','penalty_type','arm','dimension','subgroup']):
        values=g.set_index('resolution').auroc
        add(budget,penalty,'5m_minus_15m',dim,sub,'resolution_contrast','delta_5m_minus_15m_'+arm,'5m','15m',values['5m']-values['15m'])
    correlations=[]
    for (budget,penalty,res),g in p.groupby(['budget','penalty_type','resolution']):
        wide=g.pivot(index='user_id',columns='arm',values='score').loc[ids]
        for dimension,subgroup,mask in groups:
            for arm in wide.columns:
                rho=float(spearmanr(wide[arm].to_numpy()[mask],wide.H_clock.to_numpy()[mask]).statistic)
                correlations.append({'budget':budget,'penalty_type':penalty,'resolution':res,'arm':arm,'reference_arm':'H_clock','dimension':dimension,'subgroup':subgroup,'n_allocations':1,'rho_mean':rho,'rho_sd':np.nan,'rho_min':rho,'rho_max':rho,'rho_alloc_0':rho,'rho_alloc_1':np.nan,'rho_alloc_2':np.nan})
    # All input checks precede writes; original prediction/selection files are immutable.
    pd.DataFrame(correlations).to_csv(OUT/'spearman_rank_correlations.csv',index=False)
    direct.to_csv(OUT/'subgroup_metrics_per_allocation.csv',index=False)
    summary.to_csv(OUT/'subgroup_summary.csv',index=False)
    pd.DataFrame(contrast).to_csv(OUT/'paired_contrasts.csv',index=False)
    info.assign(**labels).to_csv(OUT/'reconciled_test_recording_descriptors.csv',index_label='user_id')
    sources=[pred_path,OUT/'benchmark_metrics.csv',ART/'cache_5min/customers.csv',ART/'apple_10_allocations.parquet',cuts_path]
    provenance={'script':str(Path(__file__).relative_to(ROOT)),'allocations':[0],'configurations':96,'participants':1671,'classes':{'0':579,'1':1092},'subgroups':31,'summary_rows':len(summary),'contrast_rows':len(contrast),'bootstrap':'Not generated by this script; no confidence intervals in these summaries.','interpretation':'Allocation 0 only; means/min/max equal its observed metric. Cross-allocation SD and absent allocation columns are missing. Paired effects compare refitted predictors on identical participants.','cutpoints':'Saved training cutpoints; history/occupancy recomputed from training customer metadata. Continuity/occupancy variability cutpoint provenance is inherited, not recomputed from all training caches. Test descriptors recomputed from full-history masks/dates.','sources_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}}
    (OUT/'RECONCILIATION_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:provenance[k] for k in ['allocations','configurations','participants','subgroups','summary_rows','contrast_rows']}))

if __name__=='__main__':main()
