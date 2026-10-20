"""Small source-only Apple/Garmin transfer pilot from saved first-40-day caches.
Run with /Users/bulat/micromamba/envs/datenspende/bin/python run_transfer.py.
No SQL, MultiRocket transforms, target tuning or probability calibration.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sklearn

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCES={'Apple':ROOT/'experiments/apple_cohort_scaling/artifacts',
         'Garmin':ROOT/'experiments/garmin_temporal_binning/artifacts'}
ALLOCATIONS={'Apple':'apple_10_allocations.parquet','Garmin':'garmin_binning_5_allocations.parquet'}
ALPHAS=[100,500,1000,5000,10000]
SEED=20261019
HR=['mean','median','sd','vmin','vmax']
REC=['n','cov_h','hours']
FEATURES={'HR':[f'{s}_{agg}' for agg in ['mean','sd'] for s in HR],
          'recording':[f'{s}_{agg}' for agg in ['mean','sd'] for s in REC]+[f'occ_hour_{h:02d}' for h in range(24)]}
FEATURES['combined']=FEATURES['HR']+FEATURES['recording']+['weekday_minus_weekend_hr']


def extract(device):
    art=SOURCES[device]
    alloc=pd.read_parquet(art/ALLOCATIONS[device]);alloc['customer_id']=alloc.customer_id.astype(str)
    customers=pd.read_csv(art/'cache_5min/customers.csv',dtype={'customer_id':str})
    meta=alloc.merge(customers[['customer_id','file','n_days']],on='customer_id',validate='one_to_one',suffixes=('_allocation',''))
    meta=meta.loc[meta.n_days>=40].copy().sort_values('customer_id')
    rows=[];digest=hashlib.sha256()
    for row in meta.itertuples():
        f=art/'cache_5min'/row.file
        with np.load(f) as z:
            dates=z['date'][:40].copy();stats=z['stats'][:40].copy();hr=z['hr'][:40].copy()
        assert len(dates)==40 and np.all(np.diff(dates)>0) and stats.shape==(40,8) and hr.shape==(40,288)
        assert np.isfinite(stats).all()
        digest.update(str(row.customer_id).encode());digest.update(dates.tobytes());digest.update(stats.tobytes());digest.update(hr.tobytes())
        mean=stats.mean(axis=0);sd=stats.std(axis=0,ddof=1)
        values={f'{s}_{agg}':float(array[i]) for agg,array in [('mean',mean),('sd',sd)] for i,s in enumerate(HR+REC)}
        values.update({f'occ_hour_{h:02d}':float(v) for h,v in enumerate(np.isfinite(hr).reshape(40,24,12).mean(axis=(0,2)))})
        # numpy epoch day zero is Thursday; Monday=0.
        weekend=(dates+3)%7>=5
        values['weekday_minus_weekend_hr']=float(stats[~weekend,0].mean()-stats[weekend,0].mean()) if weekend.any() and (~weekend).any() else np.nan
        rows.append({'device':device,'user_id':str(row.customer_id),'y':int(row.y),'age_group':row.age_group,'split':row.alloc_0,'n_days_full':int(row.n_days),'first_selected_date':int(dates[0]),'last_selected_date':int(dates[-1]),**values})
    frame=pd.DataFrame(rows)
    assert len(frame)=={'Apple':10214,'Garmin':979}[device]
    return frame,digest.hexdigest()


def bootstrap(y,local,imported):
    rng=np.random.default_rng(SEED);indices=[np.flatnonzero(y==c) for c in [0,1]];draws=[];local_draws=[];imported_draws=[]
    for _ in range(1000):
        ix=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in indices])
        local_auc=roc_auc_score(y[ix],local[ix]);imported_auc=roc_auc_score(y[ix],imported[ix])
        draws.append(imported_auc-local_auc);local_draws.append(local_auc);imported_draws.append(imported_auc)
    return np.quantile(draws,[.025,.975]),np.quantile(local_draws,[.025,.975]),np.quantile(imported_draws,[.025,.975])


def main():
    started=time.time();frames=[];cache_hashes={}
    for device in SOURCES:
        f,h=extract(device);frames.append(f);cache_hashes[device]=h
    data=pd.concat(frames,ignore_index=True)
    assert data.groupby('user_id').device.nunique().max()==1
    data.to_csv(OUT/'participant_features.csv',index=False)
    selection=[];predictions=[];coefficients=[]
    for mode,source in [('full','Apple'),('full','Garmin'),('size_matched','Apple'),('size_matched','Garmin')]:
        d=data.query('device == @source');train=d.query("split=='train'");val=d.query("split=='val'")
        if mode=='size_matched' and source=='Apple':
            # One prespecified draw: match source sample sizes, retain Apple class proportions.
            for split in ['train','val']:
                pool=d.query('split==@split')
                n=len(data.query("device=='Garmin' and split==@split"))
                chosen,_=train_test_split(pool,train_size=n,stratify=pool.y,random_state=SEED)
                if split=='train':train=chosen.sort_values('user_id')
                else:val=chosen.sort_values('user_id')
        pd.concat([train,val]).assign(mode=mode).to_csv(OUT/f'fit_participants_{mode}_{source}.csv',index=False)
        for block,columns in FEATURES.items():
            fitted=[]
            for alpha in ALPHAS:
                model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeClassifier(alpha=alpha))
                model.fit(train[columns],train.y)
                auc=roc_auc_score(val.y,model.decision_function(val[columns]))
                fitted.append((alpha,auc,model));selection.append({'mode':mode,'source_device':source,'block':block,'alpha':alpha,'val_auc':auc,'n_train':len(train),'n_val':len(val)})
            best=max(x[1] for x in fitted)
            alpha,auc,model=max((x for x in fitted if np.isclose(x[1],best,atol=1e-12)),key=lambda x:x[0])
            for row in selection:
                if row['mode']==mode and row['source_device']==source and row['block']==block:row['selected']=row['alpha']==alpha
            for feature,coef in zip(columns,np.asarray(model[-1].coef_).ravel()):coefficients.append({'mode':mode,'source_device':source,'block':block,'alpha':alpha,'feature':feature,'coefficient_per_source_sd':float(coef)})
            for target in SOURCES:
                test=data.query("device==@target and split=='test'").copy()
                test['mode']=mode;test['score']=model.decision_function(test[columns]);test['source_device']=source;test['target_device']=target;test['block']=block;test['alpha']=alpha
                predictions.append(test[['mode','source_device','target_device','block','alpha','user_id','y','age_group','score']])
    pred=pd.concat(predictions,ignore_index=True)
    assert not pred.duplicated(['mode','source_device','target_device','block','user_id']).any()
    metrics=[];effects=[]
    for (mode,target,block),g in pred.groupby(['mode','target_device','block']):
        w=g.pivot(index='user_id',columns='source_device',values='score');meta=g.drop_duplicates('user_id').set_index('user_id').loc[w.index]
        assert not w.isna().any().any()
        imported=next(x for x in SOURCES if x!=target)
        groups=[('all',np.ones(len(w),bool))]+[(age,(meta.age_group==age).to_numpy()) for age in sorted(meta.age_group.unique())]
        for group,mask in groups:
            y=meta.y.to_numpy()[mask];n0=int((y==0).sum());n1=int((y==1).sum())
            for source in SOURCES:
                metrics.append({'mode':mode,'source_device':source,'target_device':target,'block':block,'age_group':group,'n_test':len(y),'n_class_0':n0,'n_class_1':n1,'test_auc':roc_auc_score(y,w[source].to_numpy()[mask]) if min(n0,n1)>0 else np.nan})
            if group=='all':
                local=w[target].to_numpy();foreign=w[imported].to_numpy();(lo,hi),(local_lo,local_hi),(imported_lo,imported_hi)=bootstrap(y,local,foreign)
                effects.append({'mode':mode,'target_device':target,'block':block,'imported_source':imported,'n_test':len(y),'local_auc':roc_auc_score(y,local),'imported_auc':roc_auc_score(y,foreign),'delta_imported_minus_local':roc_auc_score(y,foreign)-roc_auc_score(y,local),'ci_lower':lo,'ci_upper':hi,'local_auc_ci_lower':local_lo,'local_auc_ci_upper':local_hi,'imported_auc_ci_lower':imported_lo,'imported_auc_ci_upper':imported_hi})
    pd.DataFrame(selection).to_csv(OUT/'alpha_selection.csv',index=False)
    pred.to_csv(OUT/'test_predictions.csv',index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'test_metrics.csv',index=False)
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'paired_transfer_effects.csv',index=False)
    counts=data.groupby(['device','split']).agg(n=('y','size'),class_1=('y','sum')).reset_index();counts['class_0']=counts.n-counts.class_1;counts.to_csv(OUT/'cohort_counts.csv',index=False)
    sources=[SOURCES[d]/f for d in SOURCES for f in [ALLOCATIONS[d],'cache_5min/customers.csv']]
    prov={'script':str(Path(__file__).relative_to(ROOT)),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'library_versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'positive_class':'recorded salutation20 -> y1; salutation10 -> y0','source_size_control':'Secondary size_matched: stratified Apple train/val downsample separately to Garmin train/val counts, preserving Apple class proportions; seed20261019; same test sets; no seed selection. Garmin model unchanged.','runtime_seconds':time.time()-started,'features':FEATURES,'eligibility':'At least 40 cached adequate days; exact earliest 40 days; existing allocation 0 splits retained','model':'Source-train median imputation, scaling, RidgeClassifier. Alpha maximizes source validation AUROC; np.isclose(atol=1e-12, rtol=1e-5) ties favor larger alpha. No train+validation refit.','evaluation':'Both models on each device use the same held-out test participants; target labels never used for selection.','bootstrap':'1000 paired class-stratified fixed-count participant resamples; seed 20261019 reset per effect; percentile 95%; conditional on fitted predictors; no multiplicity adjustment.','input_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'selected_cache_arrays_sha256':cache_hashes,'feature_csv_sha256':hashlib.sha256((OUT/'participant_features.csv').read_bytes()).hexdigest()}
    (OUT/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    print(counts.to_string(index=False));print(effects.to_string(index=False));print('runtime_seconds',prov['runtime_seconds'])

if __name__=='__main__':main()
