"""Evaluate a small fixed set of within-case temporal postprocessors."""
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import median_filter,uniform_filter1d
from scipy.special import expit,logit
from src.data import load_data
from src import cv

d=load_data();directory=Path(sys.argv[1]); scores={}
for f,y in cv.FOLDS.items():
 _,idx=cv.split(d,y);p=np.load(directory/f'pred_{f}.npy').astype(float)
 variants={'raw':p}
 for k in [3,5]:
  m=median_filter(p,size=(1,1,k),mode='nearest')
  variants[f'median{k}_half']=.5*p+.5*m
 variants['mean3_quarter']=.75*p+.25*uniform_filter1d(p,3,axis=2,mode='nearest')
 z=logit(np.clip(p,1e-10,1-1e-10))
 variants['logit_mean3_quarter']=expit(.75*z+.25*uniform_filter1d(z,3,axis=2,mode='nearest'))
 for name,v in variants.items():scores.setdefault(name,[]).append(cv.score(d,idx,v))
 print(f,{n:round(s[-1],7) for n,s in scores.items()},flush=True)
print('MEANS',{n:float(np.mean(s)) for n,s in scores.items()},flush=True)
