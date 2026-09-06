import numpy as np
class SignalQualityAgent:
    def assess(self, w):
        out={}
        for m in ["ecg","ppg","acc"]:
            x=w[m].to_numpy(float); miss=np.mean(~np.isfinite(x)); z=x[np.isfinite(x)]
            snr=20*np.log10((np.std(z)+1e-6)/(np.std(np.diff(z))+1e-6)) if len(z)>2 else 0
            validity=float(np.clip(1-miss,0,1)); artifact=float(np.clip(1/(1+np.std(np.diff(np.nan_to_num(x)))),0,1))
            q=float(np.clip(.45*validity+.3*artifact+.25*np.clip((snr+20)/40,0,1),0,1))
            out[m]={"quality_score":q,"snr":float(snr),"artifact_score":float(artifact),"missing_fraction":float(miss),"validity_score":validity}
        return out
