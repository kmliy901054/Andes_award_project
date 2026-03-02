import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import glob
import os

# -----------------------------
# 參數設定
# -----------------------------
data_folder = r"A:\DP_learn\Andes\combined_output"
csv_files = glob.glob(os.path.join(data_folder, "session_*_combined.csv"))
fixed_len = 120

data_str = """
0.7
0.85
0.9
0.6
0.9
0.92
0.9
0.45
0.4
0.72
0.88
0.52
0.75
0.48
0.15
0.3
0.35
0.75
0.85
0.9
0.92
0.3
0.32
0.8
0.85
0.85
0.9
0.3
0.55
0.88
0.85
0.9
0.7
0.75
0.85
0.2
0.31
0.77
0.8
0.8
0.85
0.65
0.66
0.75
0.8
0.8
0.77
0.85
0.9
0.6
0.8
0.1
0.91
0.28
0.17
0.83
0.86
0.12
0.17
0.89
0.3
0.91
0.7
0.46
0.38
0.6
0.95
0.5
0.86
0.78
0.97
0.95
0.93
0.82
0.91
0.88
0.06
0.14
0.16
0.26
0.89
0.84
0.72
0.8
0.87
0.31
0.21
0.56
0.31
0.88
0.6
0.81
0.65
0.83
0.5
0.6
0.66
0.53
0.91
0.95
0.92
0.88
0.7
0.46
0.73
0.82
0.92
0.89
0.3
0.49
0.93
0.88
0.71
0.76
0.83
0.6
0.79
0.2
0.35
0.51
0.86
0.4
0.96
0.88
0.85
0.78
0.81
0.92
0.91
0.88
0.84
0.52
0.23
0.68
0.52
0.51
0.47
0.22
0.08
"""

# 轉成一維 list
values = [
    float(x.strip()) for x in data_str.replace("\n", ",").split(",") if x.strip() != ""
]


if len(csv_files) == 0:
    raise FileNotFoundError(f"找不到 CSV，請確認 {data_folder}")

# -----------------------------
# 讀 CSV 並補零 + 正規化
# -----------------------------
sequences = []
target_scores = values

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
 
    df.fillna(method='ffill', inplace=True) # 向前填充
    df.fillna(0, inplace=True)
    
    keypoints = df.iloc[:, 1:].values.astype(np.float32)  # 去掉 timestamp
    num_features = keypoints.shape[1]
    seq_len = keypoints.shape[0]

    # MinMax 正規化到 [0,1]
    keypoints_min = keypoints.min(axis=0)
    keypoints_max = keypoints.max(axis=0)
    keypoints = (keypoints - keypoints_min) / (keypoints_max - keypoints_min + 1e-8)

    # 補零
    padded = np.zeros((fixed_len, num_features), dtype=np.float32)
    if seq_len >= fixed_len:
        padded = keypoints[:fixed_len, :]
    else:
        padded[:seq_len, :] = keypoints

    sequences.append(padded)

X = np.array(sequences, dtype=np.float32)
y = np.array(target_scores, dtype=np.float32).reshape(-1, 1)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# -----------------------------
# 建立 Dataset
# -----------------------------
class PoseRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# 隨機切訓練/測試集 80/20
# -----------------------------
num_samples = len(X)
indices = np.arange(num_samples)
np.random.shuffle(indices)
split = int(0.8 * num_samples)

train_idx = indices[:split]
test_idx = indices[split:]

train_dataset = PoseRegressionDataset(X[train_idx], y[train_idx])
test_dataset = PoseRegressionDataset(X[test_idx], y[test_idx])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# -----------------------------
# LSTM 回歸模型
# -----------------------------
class PoseLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=16, num_layers=2):
        super(PoseLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseLSTMRegressor(input_size=num_features).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 降低學習率

# -----------------------------
# 訓練迴圈
# -----------------------------
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        
        if torch.isnan(batch_X).any():
            print("警告：輸入批次中發現 NaN！")
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.6f}")

# -----------------------------
# 測試
# -----------------------------
model.eval()
with torch.no_grad():
    all_preds = []
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        all_preds.extend(outputs.cpu().numpy())
    print("預測分數:", all_preds)


torch.save(model.state_dict(), 'model.pt')