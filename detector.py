import joblib,numpy as np,torch
from config import MODEL_DIR,FEATURES
from .autoencoder import Autoencoder
class BehavioralDetector:
 def __init__(self):
  self.scaler=joblib.load(MODEL_DIR/'scaler.joblib');self.iso=joblib.load(MODEL_DIR/'isolation_forest.joblib');self.info=joblib.load(MODEL_DIR/'feature_info.joblib');self.ae=Autoencoder(len(FEATURES));self.ae.load_state_dict(torch.load(MODEL_DIR/'autoencoder.pt',map_location='cpu'));self.ae.eval()
 def norm(self,x,a,b): return float(np.clip((x-a)/max(b-a,1e-9),0,1))
 def predict(self,e):
  x=np.array([[float(e[f]) for f in FEATURES]]);xs=self.scaler.transform(x);ir=-float(self.iso.decision_function(xs)[0]);iscore=self.norm(ir,self.info['iso_min'],self.info['iso_max']);t=torch.tensor(xs,dtype=torch.float32)
  with torch.no_grad(): er=float(torch.mean((t-self.ae(t))**2).item())
  ascore=self.norm(er,self.info['ae_min'],self.info['ae_max']);risk=round((.45*iscore+.55*ascore)*100,2);sev='CRITICAL' if risk>=80 else 'HIGH' if risk>=60 else 'MEDIUM' if risk>=35 else 'LOW'
  return {'risk_score':risk,'severity':sev,'is_anomaly':risk>=50,'isolation_forest_score':round(iscore,4),'autoencoder_score':round(ascore,4),'model_agreement':round(1-abs(iscore-ascore),4)}
