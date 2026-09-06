import numpy as np
class FusionAgent:
    def fuse(self, quality, rep):
        q=np.array([quality[m]["quality_score"] for m in ["ecg","ppg","acc"]]); q=q/(q.sum()+1e-8)
        return {"weights":dict(zip(["ecg","ppg","acc"],q)), "quality_weighted_mean":float(q.mean())}
