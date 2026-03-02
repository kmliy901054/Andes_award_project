# tflite_to_h.py
import sys

in_file  = "emg40_mlp_int8.tflite"
out_file = "emg40_model.h"

with open(in_file, "rb") as f:
    data = f.read()

with open(out_file, "w") as f:
    f.write("const unsigned char emg40_model[] = {\n")
    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}, ")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int emg40_model_len = {len(data)};\n")

print(f"[OK] wrote {out_file}, size={len(data)} bytes")
