import pandas as pd, numpy as np
K='data/kernel/'; E='data/extended/'
uc=pd.read_csv(K+'Unit_commitment_decisions.csv', parse_dates=['starttime'])
lab=[c for c in uc.columns if c.startswith('result_committed_')]
units=list(dict.fromkeys(c.replace('result_committed_','').rsplit('_t',1)[0] for c in lab))
Y=uc[lab].to_numpy().reshape(len(uc),len(units),168)  # case, unit, hour
price=pd.read_csv(K+'Historical_day_ahead_price_2015_2025.csv',parse_dates=['date']).set_index('date').iloc[:,0]
inflow=pd.read_csv(K+'Historical_inflow_1958_2025.csv',parse_dates=['date']).set_index('date')
vol=pd.read_csv(K+'Historical_volume_2015_2024.csv',parse_dates=['date']).set_index('date')
wv=pd.read_csv(K+'Synthetic_water_value_2015_2024.csv',parse_dates=['date']).set_index('date')
mf=pd.read_csv(E+'Constraint_min_flow.csv',parse_dates=['date']).set_index('date')
mv=pd.read_csv(E+'Constraint_min_volume.csv',parse_dates=['date']).set_index('date')
pm=pd.read_csv('prediction_mapping.csv',parse_dates=['starttime'])
# plant -> (upstream reservoir(s), downstream reservoir or None) from topology YAML connections
PLANT_UP={'Hogga':['Bandak'],'Tokke':['Vinjevatn'],'Lio':['Byrtevatn'],'Byrte':['Botnedalsvatn'],'Vinje':['Vaamarvatn'],
          'Haukeli':['Vatjern'],'Songa':['Songavatn','Bitdalsvatn'],'Kjela':['Foersvatn'],'Vesle Kjela':['Kjelavatn']}
PLANT_DOWN={'Hogga':None,'Tokke':'Bandak','Lio':None,'Byrte':'Byrtevatn','Vinje':'Vinjevatn','Haukeli':'Vinjevatn',
            'Songa':'Totak','Kjela':'Hyljelihyl','Vesle Kjela':'Foersvatn'}
plant_of=lambda u:u.rsplit('_',1)[0]
starts=uc.starttime.dt.tz_localize('UTC')
def week_price(s): return price.loc[s:s+pd.Timedelta(hours=167)].to_numpy()
P=np.stack([week_price(s) for s in starts])  # case,168
def wv_at(s,r): return wv.loc[s,r] if r else 0.0
