#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== MediaPipe Pose indices =====
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24

# ===== Utils =====
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_action_id(fname):
    m = re.search(r"action_(\d+)\.csv", fname)
    return int(m.group(1))

# ===== Load Excel GT =====
def load_lr_score(xlsx):
    df = pd.read_excel(xlsx)
    mapping = {}

    for i, row in df.iterrows():
        # 用 iloc 明確指定「前 3 欄（位置）」
        values = row.iloc[:3]

        # 若有 X 或 NaN → 無效資料
        if any((isinstance(v, str) and v.lower() == "x") or pd.isna(v) for v in values):
            continue

        try:
            L = float(values.iloc[0])
            R = float(values.iloc[1])
            C = int(values.iloc[2])
        except Exception:
            continue

        mapping[i + 1] = (L, R, C)

    if not mapping:
        raise RuntimeError("No valid rows found in LR_score.xlsx")

    return mapping


# ===== Pose Preprocess =====
def preprocess_pose(df, max_missing_ratio=0.2):
    pose = np.stack([
        df[[f"Node{i}_X", f"Node{i}_Y", f"Node{i}_Z"]].to_numpy()
        for i in range(33)
    ], axis=1)  # (T,33,3)

    # missing frame = all zero
    missing = np.isclose(pose, 0.0).all(axis=(1,2))
    if missing.mean() > max_missing_ratio:
        return None

    # forward fill
    for t in range(1, len(pose)):
        if missing[t]:
            pose[t] = pose[t-1]

    # center + scale
    centers = (pose[:,L_HIP] + pose[:,R_HIP]) / 2
    scales = np.linalg.norm(pose[:,L_SHO] - pose[:,R_SHO], axis=1)
    scales[scales < 1e-6] = 1.0

    pose = (pose - centers[:,None,:]) / scales[:,None,None]
    return pose.reshape(len(pose), -1).astype(np.float32)

# ===== Dataset =====
class PoseDataset(Dataset):
    def __init__(self, files, labels, window, stride):
        self.samples = []
        for f in files:
            sid = parse_action_id(f)
            _,_,cls = labels[sid]
            df = pd.read_csv(f)
            X = preprocess_pose(df)
            if X is None or len(X) < window:
                continue
            for s in range(0, len(X)-window+1, stride):
                self.samples.append((X[s:s+window], cls))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        X, y = self.samples[i]
        return torch.from_numpy(X), torch.tensor(y)

# ===== Model =====
class PoseLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(99, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default='./collected_data')
    ap.add_argument("--xlsx", default='./LR_score.xlsx')
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    set_seed()
    labels = load_lr_score(args.xlsx)

    files = [os.path.join(args.data_dir,f) for f in os.listdir(args.data_dir)]
    files = [f for f in files if parse_action_id(f) in labels]

    # split by file
    np.random.shuffle(files)
    split = int(len(files)*0.8)
    train_f, val_f = files[:split], files[split:]

    classes = sorted(set(labels[parse_action_id(f)][2] for f in files))
    cls_map = {c:i for i,c in enumerate(classes)}

    train_ds = PoseDataset(train_f, labels, args.window, args.stride)
    val_ds   = PoseDataset(val_f, labels, args.window, args.stride)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=128)

    model = PoseLSTM(len(cls_map))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        for X,y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X,y in val_dl:
                pred = model(X).argmax(1)
                correct += (pred==y).sum().item()
                total += len(y)
        print(f"[Pose] epoch {ep} acc={correct/total:.3f}")

    torch.save(model.state_dict(), "pose_lstm.pt")

if __name__ == "__main__":
    main()
