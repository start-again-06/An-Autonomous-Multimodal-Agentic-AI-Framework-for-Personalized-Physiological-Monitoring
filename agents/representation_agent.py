import numpy as np
class PhysiologicalRepresentationAgent:
    def encode(self,w):
        f={}
        for c in ["heart_rate","hrv","activity_level","ecg","ppg","acc"]:
            x=w[c].to_numpy(float); x=np.nan_to_num(x,nan=np.nanmedian(x))
            f[c+"_mean"]=float(np.mean(x)); f[c+"_std"]=float(np.std(x))
        return f
