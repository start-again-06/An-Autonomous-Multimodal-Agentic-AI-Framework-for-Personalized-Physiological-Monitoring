import numpy as np
from sklearn.preprocessing import StandardScaler
class RiskModel:
    def __init__(self,seed=42): self.scaler=StandardScaler(); self.w=None
    def fit(self,X,y):
        X=self.scaler.fit_transform(X); self.w=np.zeros(X.shape[1]+1)
        X=np.c_[np.ones(len(X)),X]
        for _ in range(800):
            z=np.clip(X@self.w,-20,20); pr=1/(1+np.exp(-z))
            self.w-=0.12*(X.T@(pr-y))/len(y)
        return self
    def predict(self,X):
        X=self.scaler.transform(X); X=np.c_[np.ones(len(X)),X]
        return 1/(1+np.exp(-np.clip(X@self.w,-20,20)))
