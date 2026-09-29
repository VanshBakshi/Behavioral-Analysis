import numpy as np,pandas as pd,joblib,torch
from torch import nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import *
from src.data_generator import generate_dataset
from src.autoencoder import Autoencoder
df=generate_dataset(6000,RANDOM_STATE);df.to_csv(DATA_DIR/'behavioral_data.csv',index=False);normal=df[df.label==0][FEATURES];tr,va=train_test_split(normal,test_size=.2,random_state=RANDOM_STATE);sc=StandardScaler();xt=sc.fit_transform(tr);xv=sc.transform(va);iso=IsolationForest(n_estimators=250,contamination=.08,random_state=RANDOM_STATE,n_jobs=-1).fit(xt);ae=Autoencoder(len(FEATURES));opt=torch.optim.Adam(ae.parameters(),lr=LEARNING_RATE);loss=nn.MSELoss();t=torch.tensor(xt,dtype=torch.float32);loader=torch.utils.data.DataLoader(t,batch_size=BATCH_SIZE,shuffle=True)
for ep in range(EPOCHS):
 for b in loader: opt.zero_grad();l=loss(ae(b),b);l.backward();opt.step()
with torch.no_grad(): er=torch.mean((torch.tensor(xv,dtype=torch.float32)-ae(torch.tensor(xv,dtype=torch.float32)))**2,dim=1).numpy()
ir=-iso.decision_function(xv);amin,amax=np.percentile(er,[1,99.5]);imin,imax=np.percentile(ir,[1,99.5]);joblib.dump(sc,MODEL_DIR/'scaler.joblib');joblib.dump(iso,MODEL_DIR/'isolation_forest.joblib');joblib.dump({'ae_min':float(amin),'ae_max':float(amax),'iso_min':float(imin),'iso_max':float(imax)},MODEL_DIR/'feature_info.joblib');torch.save(ae.state_dict(),MODEL_DIR/'autoencoder.pt');df.sample(10,random_state=42).drop(columns='label').to_csv(SAMPLE_DIR/'sample_events.csv',index=False);print('Training complete')
