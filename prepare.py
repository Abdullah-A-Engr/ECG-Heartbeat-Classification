from pathlib import Path
import numpy as np
import pandas as pd
import torch
from load_data import load_data
ROOT=Path(__file__).resolve().parents[1]

def prepare_data():
    train_df,test_df=load_data()
    if train_df.shape[1]!=188 or test_df.shape[1]!=188: raise ValueError('Expected 188 columns per row')
    X=train_df.iloc[:,:187].to_numpy(dtype=np.float32); y=train_df.iloc[:,187].to_numpy().astype(np.int64)
    Xtest=test_df.iloc[:,:187].to_numpy(dtype=np.float32); ytest=test_df.iloc[:,187].to_numpy().astype(np.int64)
    if not set(np.unique(y)).issubset(set(range(5))) or not set(np.unique(ytest)).issubset(set(range(5))): raise ValueError('Labels must be integers 0..4')
    rng=np.random.default_rng(42); idx=rng.permutation(len(train_df)); n_val=int(.1*len(train_df)); val_idx,train_idx=idx[:n_val],idx[n_val:]
    meta={'train_rows':len(train_idx),'val_rows':len(val_idx),'test_rows':len(ytest),'signal_min':float(X.min()),'signal_max':float(X.max()),'train_counts':np.bincount(y[train_idx],minlength=5).tolist(),'val_counts':np.bincount(y[val_idx],minlength=5).tolist(),'test_counts':np.bincount(ytest,minlength=5).tolist()}
    pd.DataFrame([meta]).to_csv(ROOT/'results'/'dataset_summary.csv',index=False)
    return (torch.tensor(X[train_idx]),torch.tensor(y[train_idx]),torch.tensor(X[val_idx]),torch.tensor(y[val_idx]),torch.tensor(Xtest),torch.tensor(ytest),meta)
if __name__=='__main__':
    *_,m=prepare_data(); print(m)
