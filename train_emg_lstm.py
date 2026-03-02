#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def parse_action_id(f):
    return int(re.search(r"action_(\d+)", f).group(1))

def load_lr_score(xlsx):
    df = pd.read_excel(xlsx)
    mapping = {}

    for i, r in df.iterrows():
        values = r.iloc[:2]  # 明確用位置

        # 有 X 或 NaN → 無效
        if any((isinstance(v, str) and v.lower() == "x") or pd.isna(v) for v in values):
            continue

        try:
            L = float(values.iloc[0])
            R = float(values.iloc[1])
        except Exception:
            continue

        mapping[i + 1] = (L, R)

    if not mapping:
        raise RuntimeError("No valid EMG labels found in LR_score.xlsx")

    return mapping

def forward_fill(x):
    for i in range(1,len(x)):
        if np.isclose(x[i],0).all():
            x[i]=x[i-1]
    return x

class EMGDataset(Dataset):
    def __init__(self, files, labels, window, stride):
        self.samples=[]
        for f in files:
            sid=parse_action_id(f)
            y=np.array(labels[sid],dtype=np.float32)
            df=pd.read_csv(f)
            X=df[["EMG_L_Norm","EMG_R_Norm"]].to_numpy(np.float32)
            X=forward_fill(X)
            for s in range(0,len(X)-window+1,stride):
                self.samples.append((X[s:s+window],y))

    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        X,y=self.samples[i]
        return torch.from_numpy(X),torch.from_numpy(y)

class EMGLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 16, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self,x):
        _,(h,_)=self.lstm(x)
        return self.fc(h[-1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default='./collected_data')
    ap.add_argument("--xlsx", default='./LR_score.xlsx')
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--window",type=int,default=40)
    ap.add_argument("--stride",type=int,default=10)
    args=ap.parse_args()

    labels=load_lr_score(args.xlsx)
    files=[os.path.join(args.data_dir,f) for f in os.listdir(args.data_dir)]
    files=[f for f in files if parse_action_id(f) in labels]

    np.random.shuffle(files)
    split=int(len(files)*0.8)
    tr,va=files[:split],files[split:]

    tr_ds=EMGDataset(tr,labels,args.window,args.stride)
    va_ds=EMGDataset(va,labels,args.window,args.stride)

    tr_dl=DataLoader(tr_ds,256,shuffle=True)
    va_dl=DataLoader(va_ds,256)

    model=EMGLSTM()
    opt=torch.optim.Adam(model.parameters(),1e-3)
    loss_fn=nn.MSELoss()

    for ep in range(args.epochs):
        model.train()
        for X,y in tr_dl:
            opt.zero_grad()
            loss=loss_fn(model(X),y)
            loss.backward()
            opt.step()

        model.eval()
        mse=0;n=0
        with torch.no_grad():
            for X,y in va_dl:
                mse+=loss_fn(model(X),y).item()*len(y)
                n+=len(y)
        print(f"[EMG] epoch {ep} mse={mse/n:.4f}")

    torch.save(model.state_dict(),"emg_lstm.pt")

if __name__=="__main__":
    main()
