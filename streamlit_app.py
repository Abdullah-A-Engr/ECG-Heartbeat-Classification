from pathlib import Path
import sys,zipfile
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from model import ECGNet
ROOT=Path(__file__).resolve().parents[1]
NAMES=['N - Normal','S - Supraventricular ectopic','V - Ventricular ectopic','F - Fusion','Q - Unknown/Paced']
@st.cache_resource
def load_model():
    model=ECGNet(); model.load_state_dict(torch.load(ROOT/'results'/'best_model.pt',map_location='cpu',weights_only=True)); model.eval(); return model
st.set_page_config(page_title='ECG Heartbeat Classifier',layout='wide')
st.title('ECG Heartbeat Classification')
st.caption('Educational machine-learning demonstration. This model is not a medical diagnostic tool.')
mode=st.radio('Input method',['Choose a test-set row','Upload a CSV with 187 signal values'],horizontal=True)
if mode=='Choose a test-set row':
    with zipfile.ZipFile(ROOT/'data'/'mitbih_test.csv.zip') as z:
        csvname=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with z.open(csvname) as f: df=pd.read_csv(f,header=None)
    idx=st.number_input('Test row index',min_value=0,max_value=len(df)-1,value=0,step=1)
    signal=df.iloc[int(idx),:187].to_numpy(dtype=np.float32)
else:
    up=st.file_uploader('Upload CSV (one row, 187 columns; no label)',type=['csv'])
    if up is None: st.info('Upload a CSV to continue.'); st.stop()
    arr=pd.read_csv(up,header=None).to_numpy(dtype=np.float32)
    if arr.size!=187: st.error(f'Expected exactly 187 values; received {arr.size}.'); st.stop()
    signal=arr.reshape(-1)
fig,ax=plt.subplots(figsize=(11,3)); ax.plot(signal); ax.set(xlabel='Sample index',ylabel='Amplitude',title='ECG beat waveform'); ax.grid(alpha=.25); st.pyplot(fig); plt.close(fig)
try:
    model=load_model()
except Exception as e:
    st.error('Trained model file not found. Run the training pipeline first.'); st.stop()
with torch.no_grad(): logits=model(torch.tensor(signal).reshape(1,-1)); probs=torch.softmax(logits,dim=1)[0].numpy(); pred=int(probs.argmax())
st.subheader('Prediction'); st.success(NAMES[pred]); st.caption('Predicted class only; probabilities are model scores, not clinical certainty.')
st.bar_chart(pd.DataFrame({'Class':NAMES,'Probability':probs}).set_index('Class'))
