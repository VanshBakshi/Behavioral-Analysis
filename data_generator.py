import numpy as np,pandas as pd
from config import FEATURES
def generate_dataset(n=6000,seed=42):
 r=np.random.default_rng(seed); users=[f'user_{i:03d}' for i in range(1,51)]; dev=[f'device_{i:03d}' for i in range(1,81)]; rows=[]
 for _ in range(n):
  v=[int(np.clip(r.normal(10.5,2.4),0,23)),int(np.clip(r.poisson(4),0,20)),int(np.clip(r.poisson(.8),0,8)),float(np.clip(r.normal(75,35),5,300)),int(np.clip(r.gamma(2,18),0,150)),float(np.clip(r.gamma(2,90),0,900)),float(np.clip(r.gamma(1.7,55),0,700)),int(np.clip(r.poisson(2),1,8)),int(np.clip(r.poisson(.25),0,3)),int(np.clip(r.poisson(.4),0,5)),int(np.clip(r.normal(120,35),20,300)),int(np.clip(r.normal(230,80),20,600)),float(np.clip(r.gamma(1.3,8),0,80)),int(np.clip(r.poisson(.1),0,2))]; a=int(r.random()<.12)
  if a:
   p=r.choice(['credential','exfil','new_device','admin'])
   if p=='credential': v[0]=int(r.choice([0,1,2,3,23]));v[2]+=int(r.integers(4,10));v[7]+=int(r.integers(3,8))
   elif p=='exfil': v[4]*=r.uniform(3,6);v[5]*=r.uniform(3,7);v[6]*=r.uniform(5,12);v[11]*=r.uniform(2,4)
   elif p=='new_device': v[7]+=int(r.integers(2,7));v[8]+=int(r.integers(3,8));v[12]+=r.uniform(100,2000)
   else: v[9]+=int(r.integers(5,15));v[10]*=r.uniform(2,4);v[11]*=r.uniform(2,5)
  rows.append([r.choice(users),r.choice(dev),*v,a])
 return pd.DataFrame(rows,columns=['user_id','device_id',*FEATURES,'label'])
