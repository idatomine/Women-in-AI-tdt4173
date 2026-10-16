"""Quick EDA baseline: LightGBM on long format (case x unit x hour). Not the final pipeline."""
import sys, time, yaml, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
exec(open('analysis/eda_load.py').read())
maxvol={r:v['max_vol'] for r,v in yaml.safe_load(open('data/extended/Tokke_Vinje_topology.yaml'))['model']['reservoir'].items()}
gens=yaml.safe_load(open('data/extended/Tokke_Vinje_topology.yaml'))['model']['generator']
N=len(uc); H=168; U=len(units)
end=starts+pd.Timedelta(days=7)
# --- case x hour features
Pm=P.mean(1,keepdims=True); Ps=P.std(1,keepdims=True)+1e-3
F={}
F['price']=P; F['price_z']=(P-Pm)/Ps; F['price_rank']=P.argsort(1).argsort(1)/167.
F['price_rel']=P/(np.abs(Pm)+1); F['price_wkmean']=np.repeat(Pm,H,1)
F['price_d24']=np.stack([pd.Series(r).rolling(24,center=True,min_periods=1).mean().to_numpy() for r in P])/(np.abs(Pm)+1)
hrs=np.arange(H); F['h']=np.broadcast_to(hrs,(N,H)); F['hod']=F['h']%24
F['doy']=np.repeat(starts.dt.dayofyear.to_numpy()[:,None],H,1); F['dow']=np.repeat(starts.dt.dayofweek.to_numpy()[:,None],H,1)
case_cols=list(F)
# --- unit x case features
G={}
for i,u in enumerate(units):
  p=plant_of(u); up=PLANT_UP[p]; dn=PLANT_DOWN[p]
  wu_e=wv.loc[end,up].mean(axis=1).to_numpy(); wu_s=wv.loc[starts,up].mean(axis=1).to_numpy()
  wd_e=wv.loc[end,dn].to_numpy() if dn else np.zeros(N)
  fill=sum(vol.loc[starts,r].to_numpy() for r in up)/sum(maxvol[r] for r in up)
  fill_dn=vol.loc[starts,dn].to_numpy()/maxvol[dn] if dn else np.zeros(N)
  infl=sum(np.array([inflow[r].loc[s:s+pd.Timedelta(hours=167)].sum()*3600/1e6 for s in starts]) for r in up)/sum(maxvol[r] for r in up)
  net=wu_e-wd_e
  G[u]=dict(wv_up_end=wu_e, wv_up_start=wu_s, wv_dn_end=wd_e, net_wv=net, net_wv_rel=net/(np.abs(Pm[:,0])+1),
            fill=fill, fill_dn=fill_dn, inflow_wk=infl, wv_trend=(wu_e-wu_s)/(np.abs(wu_s)+1))
unit_cols=list(G[units[0]])
rows=[]
X=np.zeros((N,U,H,len(case_cols)+len(unit_cols)+3),dtype=np.float32)
for i,u in enumerate(units):
  for j,c in enumerate(case_cols): X[:,i,:,j]=F[c]
  for k,c in enumerate(unit_cols): X[:,i,:,len(case_cols)+k]=G[u][c][:,None]
  X[:,i,:,-3]=i; X[:,i,:,-2]=gens[u]['p_max']; X[:,i,:,-1]=gens[u]['startcost_const']
# margin feature: price minus net water value, normalised
cols=case_cols+unit_cols+['unit','p_max','startcost']
margin=(X[...,cols.index('price')]-X[...,cols.index('net_wv')])/ (np.abs(X[...,cols.index('price_wkmean')])+1)
X=np.concatenate([X,margin[...,None]],-1); cols.append('margin')
np.save('analysis/_X.npy',X); 
def run(val_year, embargo_days=7, seed=0):
  yr=starts.dt.year.to_numpy()
  va=yr==val_year
  vstart=starts[va].min(); vend=starts[va].max()
  tr=(~va)&( (starts+pd.Timedelta(days=embargo_days)<=vstart) | (starts-pd.Timedelta(days=embargo_days)>=vend) ).to_numpy()
  Xtr=X[tr].reshape(-1,X.shape[-1]); ytr=Y[tr].reshape(-1)
  Xva=X[va].reshape(-1,X.shape[-1]); yva=Y[va].reshape(-1)
  rng=np.random.default_rng(seed); sub=rng.random(len(ytr))<0.35
  m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=127,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
  m.fit(Xtr[sub],ytr[sub],categorical_feature=[cols.index('unit')])
  p=m.predict_proba(Xva)[:,1]
  per={u:round(auc(Y[va][:,i].ravel(),p.reshape(-1,U,H)[:,i].ravel()),3) for i,u in enumerate(units)}
  return auc(yva,p), per, m
if __name__=='__main__':
  for vy in [int(a) for a in sys.argv[1:]] or [2022]:
    t=time.time(); s,per,m=run(vy); print(f'val {vy}: micro AUC {s:.4f}  ({time.time()-t:.0f}s)'); print('  per-unit',per)
    imp=pd.Series(m.booster_.feature_importance('gain'),index=cols).sort_values(ascending=False); print('  top gain', (imp/imp.sum()).round(3).head(12).to_dict())
