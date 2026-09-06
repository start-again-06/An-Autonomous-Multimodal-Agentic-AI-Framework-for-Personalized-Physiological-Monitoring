import numpy as np
def window_features(df, fs=100, window=60, overlap=.5):
    step=int(window*fs*(1-overlap)); size=int(window*fs); X=[]; y=[]; meta=[]
    for s,g in df.groupby("subject_id"):
        g=g.reset_index(drop=True)
        for st in range(0,max(1,len(g)-size+1),step):
            w=g.iloc[st:st+size]
            if len(w)<size: continue
            feats=[]
            for c in ["ecg","ppg","acc","heart_rate","hrv","activity_level"]:
                x=w[c].to_numpy(float); feats += [np.nanmean(x),np.nanstd(x),np.nanmedian(x),np.nanmean(np.abs(np.diff(np.nan_to_num(x))))]
            X.append(feats); y.append(int(w.event_label.mean()>.2)); meta.append((s,w.timestamp.iloc[0]))
    return np.asarray(X,float),np.asarray(y),meta
