"""Horizon context and start-cost regret of the LP's effective hourly margins."""
CACHE = False  # cheap to derive; avoid a large redundant disk cache

import numpy as np
from scipy.ndimage import uniform_filter1d
from src.features import load_group
from src.features.relaxed_dp import _path_values
from src.data import plant_of, week_prices

def build(data):
    a, names = load_group('relaxed_lp_best', data)
    get = lambda n: np.asarray(a[..., names.index('lpb_'+n)])
    prod, on, margin = get('production'), get('unit_on'), get('margin')
    out = {}
    for n, x in [('on',on), ('margin',margin)]:
        out[f'ldc_{n}_mean6'] = uniform_filter1d(x,6,axis=2,mode='nearest')
        out[f'ldc_{n}_mean24'] = uniform_filter1d(x,24,axis=2,mode='nearest')
        out[f'ldc_{n}_week'] = x.mean(axis=2,keepdims=True)
    out['ldc_production_week'] = prod.mean(axis=2,keepdims=True)
    out['ldc_on_min'] = on.min(axis=2,keepdims=True)
    out['ldc_on_max'] = on.max(axis=2,keepdims=True)
    out['ldc_margin_min'] = margin.min(axis=2,keepdims=True)
    out['ldc_margin_max'] = margin.max(axis=2,keepdims=True)
    out['ldc_margin_positive'] = (margin > 1e-4).mean(axis=2,keepdims=True)
    for plant in ['Tokke','Vinje']:
        u = next(i for i,n in enumerate(data.units) if plant_of(n)==plant)
        out[f'ldc_{plant}_production'] = prod[:,u:u+1]
        out[f'ldc_{plant}_margin'] = margin[:,u:u+1]
    scale = np.abs(week_prices(data)).mean(axis=1,keepdims=True)+1
    regret = np.empty_like(margin)
    for u, unit in enumerate(data.units):
        g = data.generators[unit]
        m = margin[:,u]*scale
        # At a negative margin an ON unit chooses minimum output.
        reward = m*np.where(m>=0,g['p_max'],g['p_min'])
        values = _path_values(np.stack([np.zeros_like(reward),reward],axis=2),
                   g['startcost_const']*np.array([[0,1],[1,0]]))
        regret[:,u] = np.clip((values[...,1]-values[...,0])/(scale*g['p_max']),-50,50)
    out['ldc_margin_regret'] = regret
    return out
