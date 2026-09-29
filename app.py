import pandas as pd
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from config import STATIC_DIR,DATA_DIR,FEATURES
from src.detector import BehavioralDetector
app=FastAPI(title='Behavioral Analysis AI');detector=None
class Event(BaseModel):
 user_id:str='user_001';device_id:str='device_001';hour:float=10;login_count_24h:float=4;failed_logins_24h:float=0;session_minutes:float=60;files_accessed_24h:float=20;downloads_mb_24h:float=100;uploads_mb_24h:float=50;unique_ips_24h:float=2;new_devices_7d:float=0;admin_actions_24h:float=0;process_count_24h:float=120;network_connections_24h:float=220;geo_distance_km:float=5;password_changes_30d:float=0
@app.on_event('startup')
def load():
 global detector
 try: detector=BehavioralDetector()
 except Exception as e: print('Run python train.py first:',e)
@app.get('/',response_class=HTMLResponse)
def home(): return (STATIC_DIR/'index.html').read_text()
@app.get('/health')
def health(): return {'status':'ok','models_loaded':detector is not None}
@app.get('/stats')
def stats():
 df=pd.read_csv(DATA_DIR/'behavioral_data.csv');return {'events':len(df),'anomalies':int(df.label.sum()),'users':int(df.user_id.nunique()),'devices':int(df.device_id.nunique())}
@app.post('/analyze')
def analyze(e:Event):
 if detector is None: raise HTTPException(503,'Run python train.py first')
 d=e.model_dump();r=detector.predict(d);df=pd.read_csv(DATA_DIR/'behavioral_data.csv');base=df[df.label==0][FEATURES].median();reasons=[]
 for f in FEATURES:
  if abs(float(d[f])-float(base[f]))/max(abs(float(base[f])),1)>.7: reasons.append(f.replace('_',' '))
 r.update({'user_id':e.user_id,'device_id':e.device_id,'reasons':reasons[:5]});return r
if __name__=='__main__':
 import uvicorn;uvicorn.run(app,host='127.0.0.1',port=8000)
