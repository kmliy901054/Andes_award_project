# export_emglstm_to_c.py
import torch
import numpy as np

from model import EMGLSTM  # make sure EMGLSTM exists in model.py

# ---- set these to match your model ----
input_size  = 2
hidden_size = 16
num_layers  = 1

ckpt_path = "emg_lstm.pt"      # state_dict (or full checkpoint containing state_dict)
out_path  = "model_weights.h"    # generated C header


def dump_array(f, name, arr: np.ndarray, dtype="float"):
    arr = np.asarray(arr).reshape(-1)  # flatten
    f.write(f"static const {dtype} {name}[{arr.size}] = {{\n")
    for i, v in enumerate(arr):
        f.write(f"{float(v):.9e}f,")
        if (i + 1) % 8 == 0:
            f.write("\n")
    f.write("\n};\n\n")


def load_state_dict_any(ckpt_obj):
    # supports either raw state_dict or {"state_dict": ...} style checkpoints
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        # if it already looks like a state_dict (has parameter-like keys), use it
        return ckpt_obj
    raise TypeError("Unsupported checkpoint format: expected dict-like object.")


# ---- load model ----
m = EMGLSTM().eval()

ckpt = torch.load(ckpt_path, map_location="cpu")
sd = load_state_dict_any(ckpt)
m.load_state_dict(sd, strict=True)

# ---- LSTM weights (layer 0, unidirectional) ----
# PyTorch shapes:
#  lstm.weight_ih_l0: (4H, I)
#  lstm.weight_hh_l0: (4H, H)
#  lstm.bias_ih_l0:   (4H,)
#  lstm.bias_hh_l0:   (4H,)
W_ih = sd["lstm.weight_ih_l0"].detach().cpu().numpy()
W_hh = sd["lstm.weight_hh_l0"].detach().cpu().numpy()
b_ih = sd["lstm.bias_ih_l0"].detach().cpu().numpy()
b_hh = sd["lstm.bias_hh_l0"].detach().cpu().numpy()

# ---- FC layers (fc = [Linear(16->8), ReLU, Linear(8->2), Sigmoid]) ----
FC0_W = sd["fc.0.weight"].detach().cpu().numpy()  # (8, 16)
FC0_B = sd["fc.0.bias"].detach().cpu().numpy()    # (8,)
FC2_W = sd["fc.2.weight"].detach().cpu().numpy()  # (2, 8)
FC2_B = sd["fc.2.bias"].detach().cpu().numpy()    # (2,)

with open(out_path, "w") as f:
    f.write("#pragma once\n\n")
    f.write("#include <stdint.h>\n\n")

    f.write(f"#define MODEL_INPUT_SIZE   {input_size}\n")
    f.write(f"#define MODEL_HIDDEN_SIZE  {hidden_size}\n")
    f.write(f"#define MODEL_NUM_LAYERS   {num_layers}\n")
    f.write(f"#define MODEL_BIDIR        0\n")
    f.write(f"#define MODEL_OUTPUT_SIZE  2\n\n")

    # LSTM
    dump_array(f, "LSTM0_W_IH", W_ih)
    dump_array(f, "LSTM0_W_HH", W_hh)
    dump_array(f, "LSTM0_B_IH", b_ih)
    dump_array(f, "LSTM0_B_HH", b_hh)

    # FC
    dump_array(f, "FC0_W", FC0_W)
    dump_array(f, "FC0_B", FC0_B)
    dump_array(f, "FC2_W", FC2_W)
    dump_array(f, "FC2_B", FC2_B)

print(f"Wrote {out_path}")
