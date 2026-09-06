import yaml, pandas as pd, numpy as np, os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
from preprocessing.preprocessing import window_features
from models.risk_model import RiskModel
def main():
 cfg=yaml.safe_load(open("config.yaml")); df=pd.read_csv("data/synthetic/physiological_demo.csv")
 X,y,_=window_features(df); X=np.nan_to_num(X); m=RiskModel(cfg["seed"]).fit(X,y); p=m.predict(X); pred=(p>=.5).astype(int)
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 out={"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"auroc":roc_auc_score(y,p) if len(np.unique(y))>1 else float("nan"),"auprc":average_precision_score(y,p),"sensitivity":tp/(tp+fn+1e-9),"specificity":tn/(tn+fp+1e-9)}
 os.makedirs("outputs/tables",exist_ok=True); pd.DataFrame([out]).to_csv("outputs/tables/evaluation.csv",index=False); print(pd.Series(out))
if __name__=="__main__": main()
