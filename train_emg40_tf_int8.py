#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

def parse_action_id(fname: str) -> int:
    m = re.search(r"action_(\d+)", fname)
    if not m:
        raise ValueError(f"Bad filename: {fname}")
    return int(m.group(1))

def load_lr_score(xlsx: str):
    df = pd.read_excel(xlsx)
    mapping = {}
    for i, r in df.iterrows():
        v = r.iloc[:2]
        if any((isinstance(x,str) and x.lower()=="x") or pd.isna(x) for x in v):
            continue
        try:
            mapping[i+1] = (float(v.iloc[0]), float(v.iloc[1]))
        except:
            continue
    if not mapping:
        raise RuntimeError("No valid EMG labels found in LR_score.xlsx")
    return mapping

def forward_fill_zeros(x: np.ndarray) -> np.ndarray:
    for i in range(1, len(x)):
        if np.isclose(x[i], 0).all():
            x[i] = x[i-1]
    return x

def build_xy(files, labels, window=40, stride=10):
    Xs, Ys = [], []
    for f in files:
        sid = parse_action_id(os.path.basename(f))
        if sid not in labels:
            continue
        y = np.array(labels[sid], dtype=np.float32)

        df = pd.read_csv(f)
        X = df[["EMG_L_Norm", "EMG_R_Norm"]].to_numpy(dtype=np.float32)
        X = forward_fill_zeros(X)

        if len(X) < window:
            continue
        for s in range(0, len(X)-window+1, stride):
            Xs.append(X[s:s+window])
            Ys.append(y)

    if not Xs:
        raise RuntimeError("No samples built. Check data/window/stride.")
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./collected_data")
    ap.add_argument("--xlsx", default="./LR_score.xlsx")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    labels = load_lr_score(args.xlsx)
    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)
             if f.startswith("action_") and f.endswith(".csv")]
    files = [f for f in files if parse_action_id(os.path.basename(f)) in labels]

    np.random.shuffle(files)
    split = int(len(files) * 0.8)
    tr_files, va_files = files[:split], files[split:]

    Xtr, Ytr = build_xy(tr_files, labels, window=args.window, stride=args.stride)
    Xva, Yva = build_xy(va_files, labels, window=args.window, stride=args.stride)

    # ----- MLP: Flatten(40*2=80) -> Dense -> Dense -> Dense(2,sigmoid)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(args.window, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mse"])
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
              epochs=args.epochs, batch_size=args.batch, verbose=1)

    # ----- INT8 quantization (TFLite Micro friendly)
    def representative_dataset():
        n = min(500, len(Xtr))
        for i in range(n):
            yield [Xtr[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    out_path = "emg40_mlp_int8.tflite"
    open(out_path, "wb").write(tflite_int8)
    print(f"[OK] wrote {out_path}")
    print("[Next] xxd -i emg40_mlp_int8.tflite > emg40_model.h")

if __name__ == "__main__":
    main()
