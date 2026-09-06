from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
def classification_metrics(y,p):
 q=(p>=.5).astype(int); return {'accuracy':accuracy_score(y,q),'precision':precision_score(y,q,zero_division=0),'recall':recall_score(y,q,zero_division=0),'f1':f1_score(y,q,zero_division=0),'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p)}
