from pathlib import Path
import sys,time,csv
import torch
import matplotlib.pyplot as plt
from model import ECGNet,softmax_cross_entropy
from prepare import prepare_data
ROOT=Path(__file__).resolve().parents[1]

def evaluate(model,X,y):
    model.eval()
    with torch.no_grad():
        logits=model(X); loss=float(softmax_cross_entropy(logits,y)); acc=float((logits.argmax(1)==y).float().mean())
    return loss,acc

def train(alpha,epochs=30):
    torch.manual_seed(0); model=ECGNet()
    Xtr,ytr,Xv,yv,*_=prepare_data()
    params=[model.W1,model.b1,model.W2,model.b2,model.W3,model.b3]
    velocities=[torch.zeros_like(p) for p in params]
    weights={model.W1,model.W2,model.W3}; history=[]; best=float('inf'); best_epoch=0
    for epoch in range(epochs):
        model.train(); order=torch.randperm(len(Xtr))
        for start in range(0,len(order),128):
            ids=order[start:start+128]; xb=Xtr[ids]; yb=ytr[ids]
            model.zero_grad(set_to_none=True); loss=softmax_cross_entropy(model(xb),yb); loss.backward()
            with torch.no_grad():
                for p,v in zip(params,velocities):
                    g=p.grad
                    if p in weights: g=g+1e-4*p
                    v.mul_(.9).add_(g,alpha=.1)
                    p.add_(v,alpha=-alpha)
        tl,ta=evaluate(model,Xtr,ytr); vl,va=evaluate(model,Xv,yv)
        history.append({'epoch':epoch+1,'train_loss':tl,'train_accuracy':ta,'val_loss':vl,'val_accuracy':va})
        if vl<best:
            best=vl; best_epoch=epoch+1; torch.save(model.state_dict(),ROOT/'results'/f'best_model_lr_{alpha}.pt')
        print(f'lr={alpha} epoch={epoch+1:02d} train_loss={tl:.4f} val_loss={vl:.4f}',flush=True)
    return history,best_epoch

def main():
    all_hist={}; bests={}
    for lr in [0.01,0.1,0.5]:
        h,e=train(lr); all_hist[lr]=h; bests[lr]=min(x['val_loss'] for x in h)
    chosen=min(bests,key=bests.get); hist=all_hist[chosen]; best_epoch=min(hist,key=lambda x:x['val_loss'])['epoch']
    torch.save(torch.load(ROOT/'results'/f'best_model_lr_{chosen}.pt',map_location='cpu',weights_only=True),ROOT/'results'/'best_model.pt')
    import pandas as pd
    for lr,h in all_hist.items(): pd.DataFrame(h).to_csv(ROOT/'results'/f'history_lr_{lr}.csv',index=False)
    pd.DataFrame([{'learning_rate':lr,'lowest_val_loss':bests[lr],'val_accuracy_at_min_loss':min(all_hist[lr],key=lambda x:x['val_loss'])['val_accuracy']} for lr in all_hist]).to_csv(ROOT/'results'/'learning_rate_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    for lr,h in all_hist.items(): ax.plot([x['epoch'] for x in h],[x['val_loss'] for x in h],label=f'lr={lr}')
    ax.set(xlabel='Epoch',ylabel='Validation loss',title='Validation loss by learning rate'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(ROOT/'results/figures/learning_rate_comparison.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot([x['epoch'] for x in hist],[x['train_loss'] for x in hist],label='Train'); ax.plot([x['epoch'] for x in hist],[x['val_loss'] for x in hist],label='Validation'); ax.set(xlabel='Epoch',ylabel='Loss',title=f'Loss curves (lr={chosen})'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(ROOT/'results/figures/loss_curves.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot([x['epoch'] for x in hist],[x['train_accuracy'] for x in hist],label='Train'); ax.plot([x['epoch'] for x in hist],[x['val_accuracy'] for x in hist],label='Validation'); ax.set(xlabel='Epoch',ylabel='Accuracy',title=f'Accuracy curves (lr={chosen})'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(ROOT/'results/figures/accuracy_curves.png',dpi=160); plt.close(fig)
    print('CHOSEN',chosen,'BEST_EPOCH',best_epoch)
if __name__=='__main__': main()
