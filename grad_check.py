import torch
from model import ECGNet,softmax_cross_entropy

def run_gradient_check(X,y):
    torch.manual_seed(0); model=ECGNet(); logits,_,a2=model.forward_with_activations(X[:64]); loss=softmax_cross_entropy(logits,y[:64]); loss.backward()
    probs=torch.softmax(logits.detach(),dim=1); onehot=torch.zeros_like(probs); onehot[torch.arange(len(y[:64])),y[:64]]=1.0; dlogits=(probs-onehot)/len(y[:64])
    manual_W=a2.detach().T@dlogits; manual_b=dlogits.sum(dim=0)
    return {'loss':float(loss.detach()),'W3_max_abs_diff':float((manual_W-model.W3.grad).abs().max()),'b3_max_abs_diff':float((manual_b-model.b3.grad).abs().max())}

if __name__=='__main__':
    import sys
    from prepare import prepare_data
    X,y,*_=prepare_data()
    print(run_gradient_check(X,y))
