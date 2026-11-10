from pathlib import Path
import torch,matplotlib.pyplot as plt
from model import ECGNet
from prepare import prepare_data
from metrics import classification_metrics
ROOT=Path(__file__).resolve().parents[1]

def main():
    *_,Xtest,ytest,meta=prepare_data(); model=ECGNet(); model.load_state_dict(torch.load(ROOT/'results'/'best_model.pt',map_location='cpu',weights_only=True)); model.eval()
    with torch.no_grad(): pred=model(Xtest).argmax(1)
    m=classification_metrics(ytest,pred); m['always_N_baseline']=meta['test_counts'][0]/meta['test_rows']
    import json
    (ROOT/'results'/'test_metrics.json').write_text(json.dumps(m,indent=2))
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(m['confusion_matrix'],cmap='Blues'); ax.set_xticks(range(5),['N','S','V','F','Q']); ax.set_yticks(range(5),['N','S','V','F','Q']); ax.set(xlabel='Predicted',ylabel='Actual',title='Test confusion matrix')
    for i,row in enumerate(m['confusion_matrix']):
        for j,v in enumerate(row): ax.text(j,i,str(v),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(ROOT/'results/figures/confusion_matrix.png',dpi=160); plt.close(fig)
    print(json.dumps(m,indent=2))
if __name__=='__main__': main()
