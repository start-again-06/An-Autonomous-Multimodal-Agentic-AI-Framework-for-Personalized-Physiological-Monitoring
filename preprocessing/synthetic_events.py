import numpy as np, pandas as pd
def generate_dataset(cfg, out_csv):
    rng=np.random.default_rng(cfg["seed"]); fs=cfg["synthetic"]["sampling_rate"]
    n=cfg["synthetic"]["subjects"]; secs=cfg["synthetic"]["duration_minutes"]*60
    rows=[]
    for s in range(n):
        t=np.arange(0,secs,1/fs); phase=2*np.pi*t/86400
        activity=np.clip(0.15+0.35*np.sin(2*np.pi*t/180)+rng.normal(0,.08,len(t)),0,1)
        hr0=68+rng.normal(0,4); hr=hr0+18*activity+2*np.sin(2*np.pi*t/300)+rng.normal(0,1.2,len(t))
        event_start=int(secs*.62); event_dur=int(rng.integers(60,121)); precursor=60
        event=np.zeros(len(t),int); a=max(0,event_start-precursor); b=min(len(t),int((event_start+event_dur)*fs))
        event[int(event_start*fs):b]=1
        pre=(t>=a/fs)&(t<event_start)
        ev=(t>=event_start)&(t<b/fs)
        sev=rng.uniform(.45,.9)
        hr[pre]+=sev*5; hr[ev]+=sev*18
        hrv=55-0.35*(hr-70)+rng.normal(0,3,len(t)); hrv[ev]-=20*sev
        # simplified ECG/PPG proxies at physiological frequencies
        ecg=np.sin(2*np.pi*(hr/60)*t)+0.04*rng.normal(size=len(t))
        ppg=(1+0.15*np.sin(2*np.pi*t/2.5))*np.sin(2*np.pi*(hr/60)*t-0.35)+0.03*rng.normal(size=len(t))
        ppg[ev]*=(1-0.18*sev)
        acc=activity+rng.normal(0,.04,len(t))
        if rng.random()<cfg["synthetic"]["artifact_probability"]:
            ix=rng.choice(len(t),size=max(1,int(.01*len(t))),replace=False); ppg[ix]+=rng.normal(0,.8,len(ix))
        miss=rng.random(len(t))<cfg["synthetic"]["missing_probability"]
        ecg[miss]=np.nan; ppg[miss]=np.nan
        df=pd.DataFrame({"subject_id":s,"timestamp":t,"ecg":ecg,"ppg":ppg,"acc":acc,
                         "heart_rate":hr,"hrv":hrv,"activity_level":activity,"event_label":event,
                         "event_id":np.where(event, f"S{s}_E0","")})
        rows.append(df)
    out=pd.concat(rows,ignore_index=True); out.to_csv(out_csv,index=False); return out
