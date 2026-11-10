import math
import torch
from torch import nn

class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1=nn.Parameter(torch.empty(187,128)); self.b1=nn.Parameter(torch.zeros(128))
        self.W2=nn.Parameter(torch.empty(128,64)); self.b2=nn.Parameter(torch.zeros(64))
        self.W3=nn.Parameter(torch.empty(64,5)); self.b3=nn.Parameter(torch.zeros(5))
        with torch.no_grad():
            self.W1.normal_(0,math.sqrt(2/187)); self.W2.normal_(0,math.sqrt(2/128)); self.W3.normal_(0,math.sqrt(2/64))
    def forward_with_activations(self,x):
        a1=torch.clamp(x@self.W1+self.b1,min=0)
        a2=torch.clamp(a1@self.W2+self.b2,min=0)
        logits=a2@self.W3+self.b3
        return logits,a1,a2
    def forward(self,x): return self.forward_with_activations(x)[0]

def softmax_cross_entropy(logits, labels):
    rows=torch.arange(labels.shape[0],device=labels.device)
    return (torch.logsumexp(logits,dim=1)-logits[rows,labels]).mean()
