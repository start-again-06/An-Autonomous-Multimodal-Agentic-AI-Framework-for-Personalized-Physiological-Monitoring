import numpy as np
class PersonalizationAgent:
    def __init__(self): self.baselines={}
    def update(self,sid,features,alpha=.05):
        hr=features["heart_rate_mean"]; hrv=features["hrv_mean"]
        old=self.baselines.get(sid,{"hr":hr,"hrv":hrv})
        old["hr"]=(1-alpha)*old["hr"]+alpha*hr; old["hrv"]=(1-alpha)*old["hrv"]+alpha*hrv
        self.baselines[sid]=old; return old
    def deviation(self,features,b):
        return float(np.sqrt(((features["heart_rate_mean"]-b["hr"])/8)**2+((features["hrv_mean"]-b["hrv"])/12)**2))
