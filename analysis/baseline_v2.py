"""EDA baseline v2: adds physics features (price vs +7d water value, scarcity coefficient)."""
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
X0=np.load('analysis/_X.npy')
exec(open('analysis/eda_load.py').read())
N,U,H=Y.shape
end=starts+pd.Timedelta(days=7)
prh=price.resample('h').mean()
P2=np.stack([prh.loc[s+pd.Timedelta(days=7):s+pd.Timedelta(hours=7*24+167)].to_numpy() for s in starts])  # week after horizon
Pm=P.mean(1); P2m=P2.mean(1)
new=[]
for i,u in enumerate(units):
  p=plant_of(u); up=PLANT_UP[p]; dn=PLANT_DOWN[p]
  wu=wv.loc[end,up].mean(axis=1).to_numpy(); wus=wv.loc[starts,up].mean(axis=1).to_numpy()
  wd=wv.loc[end,dn].to_numpy() if dn else np.zeros(N)
  coef_s=wus/(np.abs(Pm)+1e-3); coef_e=wu/(np.abs(P2m)+1e-3)
  marg=P-wu[:,None]; scale=np.abs(P).mean(1,keepdims=True)+1
  f=np.stack([marg/scale, P/(np.abs(wu)[:,None]+1), (P-(wu-wd)[:,None])/scale,
              np.repeat(coef_s[:,None],H,1), np.repeat(coef_e[:,None],H,1),
              np.repeat(((marg>0).mean(1))[:,None],H,1),  # share of profitable hours in week
              np.repeat((np.clip(marg,0,None).sum(1)/scale[:,0]/H)[:,None],H,1),
              np.repeat((P2m/(np.abs(Pm)+1))[:,None],H,1)],-1)
  new.append(f)
X=np.concatenate([X0,np.stack(new,1).astype(np.float32)],-1)
def run(vy,seed=0,emb=7):
  va=(starts.dt.year==vy).to_numpy(); v0=starts[va].min(); v1=starts[va].max()
  tr=(~va)&((starts+pd.Timedelta(days=emb)<=v0)|(starts-pd.Timedelta(days=emb)>=v1)).to_numpy()
  Xt=X[tr].reshape(-1,X.shape[-1]); yt=Y[tr].ravel(); sub=np.random.default_rng(seed).random(len(yt))<0.35
  m=lgb.LGBMClassifier(n_estimators=800,learning_rate=0.05,num_leaves=127,min_child_samples=200,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
  m.fit(Xt[sub],yt[sub],categorical_feature=[X0.shape[-1]-4])
  p=m.predict_proba(X[va].reshape(-1,X.shape[-1]))[:,1].reshape(-1,U,H)
  return auc(Y[va].ravel(),p.ravel()),p,va
for vy in [int(a) for a in sys.argv[1:]]:
  t=time.time(); s,p,va=run(vy)
  yv=Y[va]
  # oracle-level diagnostic: replace prediction with true weekly ON fraction (level-only)
  lvl=np.broadcast_to(yv.mean(2,keepdims=True),yv.shape)
  s_lvl=auc(yv.ravel(),lvl.ravel())
  # level+our timing: rank-within-unit-week from model plus true level
  print(f'val {vy}: v2 AUC {s:.4f}  | oracle-level-only {s_lvl:.4f}  ({time.time()-t:.0f}s)')
  print('   per-unit',{u:round(auc(yv[:,i].ravel(),p[:,i].ravel()),3) for i,u in enumerate(units)})
