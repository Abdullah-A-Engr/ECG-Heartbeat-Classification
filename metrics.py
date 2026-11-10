import torch

def classification_metrics(y_true,y_pred,num_classes=5):
    cm=torch.zeros((num_classes,num_classes),dtype=torch.int64)
    for t,p in zip(y_true.reshape(-1),y_pred.reshape(-1)): cm[int(t),int(p)]+=1
    per=[]
    for c in range(num_classes):
        denom=cm[c].sum().item(); per.append(float(cm[c,c].item()/denom) if denom else 0.0)
    acc=float(cm.diag().sum().item()/cm.sum().item())
    return {'accuracy':acc,'balanced_accuracy':sum(per)/num_classes,'per_class_accuracy':per,'confusion_matrix':cm.tolist()}
