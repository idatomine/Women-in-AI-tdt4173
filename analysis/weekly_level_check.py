"""Year-held-out weekly-fraction predictor; no validation labels used in fitting."""
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.special import expit, logit
from src.data import load_data
from src.features import build_matrix
from src.run import load_config
from src import cv

def main():
    import sys
    cfg=load_config(sys.argv[1]);d=load_data()
    X,names=build_matrix(cfg['features'],d,cfg.get('drop_features',[]))
    # Summaries are allowed inputs from each case's own horizon.
    weekly=np.concatenate([X.mean(axis=2),X.std(axis=2),X.min(axis=2),X.max(axis=2)],axis=-1)
    del X
    cols=[f'{stat}_{n}' for stat in ['mean','std','min','max'] for n in names]
    results={w:[] for w in [0,.25,.5,1.]}
    for f,year in cv.FOLDS.items():
        train,val=cv.split(d,year)
        model=lgb.LGBMRegressor(n_estimators=500,learning_rate=.03,num_leaves=31,min_child_samples=50,
              colsample_bytree=.8,random_state=0,n_jobs=4,deterministic=True,force_row_wise=True,verbosity=-1)
        model.fit(weekly[train].reshape(-1,len(cols)),d.Y[train].mean(axis=2).ravel(),feature_name=cols,
                  categorical_feature=['mean_unit'])
        level=np.clip(model.predict(weekly[val].reshape(-1,len(cols))).reshape(len(val),-1),.001,.999)
        p=np.load(Path('experiments')/cfg['name']/f'pred_{f}.npy')
        # Shift each week coherently, retaining hourly ranking and a common global strength.
        offset=logit(level)-logit(np.clip(p.mean(axis=2),.001,.999))
        for w in results:
            pred=expit(logit(np.clip(p,1e-7,1-1e-7))+w*offset[...,None])
            results[w].append(cv.score(d,val,pred))
        print(f, {w:round(s[-1],7) for w,s in results.items()},flush=True)
    print('MEANS',{w:float(np.mean(s)) for w,s in results.items()},flush=True)
if __name__=='__main__':main()
