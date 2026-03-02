import time, json, threading
from collections import deque

import numpy as np
import asyncio
from bleak import BleakScanner, BleakClient
import cv2

import torch
import torch.nn as nn

from mediapipe import solutions as mp_solutions

# ======================
# User config
# ======================
EMG_SOURCE = "BLE"  # "BLE" or "SERIAL"

# ---- BLE (default) ----
BLE_DEVICE_NAME = "CorvetteT1-EMG"  # BLE.setLocalName(...)
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_UUID      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # notify
SCAN_TIMEOUT_SEC  = 8.0

# ---- SERIAL (optional fallback) ----
SERIAL_PORT = "COM4"          # <- 改成你的 Corvette T1 port
BAUD = 115200
CAMERA_INDEX = 0              # <- 0/1
POSE_MODEL_PATH = "pose_lstm.pt"

# Pose model settings (match your training)
NUM_CLASSES = 4               # 你的 xlsx label 是 0..3
POSE_DIM = 99                 # 33*3
POSE_TARGET_FRAMES = 110      # 你說一個完整動作=110 frames

# Integrator / trigger settings
SPEED_JOINTS = [15, 16, 11, 12, 23, 24]   # wrists/shoulders/hips
START_SPEED_TH = 0.010     # 動作開始門檻(越大越不敏感)
STOP_SPEED_TH  = 0.006     # 動作結束門檻(越小越容易結束)
STOP_HOLD_FRAMES = 12      # 連續多少 frame 低於 STOP 才算結束

# EMG buffer seconds (keep last N seconds)
EMG_KEEP_SEC = 15.0

# ======================
# Pose preprocessing (match training idea)
# ======================
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24

def preprocess_pose_seq(pose_xyz_seq: np.ndarray) -> np.ndarray:
    """
    pose_xyz_seq: (T,33,3) in mediapipe normalized coords
    - forward fill missing frames (missing = all zeros)
    - center: hip midpoint
    - scale: shoulder distance
    return: (T, 99) float32
    """
    T = pose_xyz_seq.shape[0]
    pose = pose_xyz_seq.copy()

    missing = np.isclose(pose, 0.0).all(axis=(1,2))
    for t in range(1, T):
        if missing[t]:
            pose[t] = pose[t-1]

    centers = (pose[:, L_HIP] + pose[:, R_HIP]) / 2.0
    scales = np.linalg.norm(pose[:, L_SHO] - pose[:, R_SHO], axis=1)
    scales[scales < 1e-6] = 1.0

    pose = (pose - centers[:, None, :]) / scales[:, None, None]
    return pose.reshape(T, 99).astype(np.float32)

def resample_to_fixed_length(seq_2d: np.ndarray, target_len: int) -> np.ndarray:
    """
    seq_2d: (T,D)
    return (target_len, D) using linear interpolation over time
    """
    T, D = seq_2d.shape
    if T == target_len:
        return seq_2d.astype(np.float32)
    if T < 2:
        out = np.repeat(seq_2d[:1], target_len, axis=0)
        return out.astype(np.float32)

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(x_new, x_old, seq_2d[:, d])
    return out

def pose_motion_energy(pose_xyz: np.ndarray, prev_pose_xyz: np.ndarray) -> float:
    """
    pose_xyz, prev_pose_xyz: (33,3)
    compute avg speed over selected joints
    """
    if prev_pose_xyz is None:
        return 0.0
    v = 0.0
    c = 0
    for j in SPEED_JOINTS:
        a = pose_xyz[j]
        b = prev_pose_xyz[j]
        if np.isclose(a, 0.0).all() or np.isclose(b, 0.0).all():
            continue
        v += float(np.linalg.norm(a - b))
        c += 1
    return v / max(c, 1)

# ======================
# Pose model definition (match your PyTorch PoseLSTM)
# ======================
class PoseLSTM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(POSE_DIM, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def load_pose_model(path: str) -> nn.Module:
    model = PoseLSTM(NUM_CLASSES)
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

# ======================
# EMG reader thread
# ======================
# board line: Lp,Rp,imbalance,magnitude,finalL,finalR
def parse_emg_line(line: str):
    """Parse one CSV line from EMG board.

    Accepts either:
      - 6 cols: Lp,Rp,imbalance,magnitude,finalL,finalR
      - 7 cols: tmillis,Lp,Rp,imbalance,magnitude,finalL,finalR
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) == 7:
        parts_vals = parts[1:]
        try:
            tmillis = float(parts[0])
        except:
            tmillis = None
    elif len(parts) == 6:
        tmillis = None
        parts_vals = parts
    else:
        return None

    try:
        vals = list(map(float, parts_vals))
        out = {
            "t_pc": time.time(),  # PC time for sync
            "Lp": vals[0],
            "Rp": vals[1],
            "imbalance": vals[2],
            "magnitude": vals[3],
            "finalL": vals[4],
            "finalR": vals[5],
        }
        if tmillis is not None:
            out["tmillis"] = tmillis
        return out
    except:
        return None

class EMGBuffer:
    def __init__(self, keep_sec=15.0):
        self.keep_sec = keep_sec
        self.buf = deque()  # each item dict with t_pc
        self.lock = threading.Lock()

    def add(self, item):
        with self.lock:
            self.buf.append(item)
            self._gc_locked()

    def _gc_locked(self):
        now = time.time()
        while self.buf and (now - self.buf[0]["t_pc"] > self.keep_sec):
            self.buf.popleft()

    def slice_by_time(self, t0, t1):
        with self.lock:
            self._gc_locked()
            return [x for x in self.buf if (t0 <= x["t_pc"] <= t1)]

async def _has_uart_service(adv) -> bool:
    """Check advertised service UUIDs for Nordic UART Service UUID."""
    uuids = getattr(adv, "service_uuids", None) or []
    uuids = [u.lower() for u in uuids]
    return UART_SERVICE_UUID.lower() in uuids


async def _ble_find_device(timeout: float):
    """Prefer finding the device by advertised UART service UUID; fall back to name."""
    found = await BleakScanner.discover(timeout=timeout, return_adv=True)
    # found: dict[address] -> (BLEDevice, AdvertisementData)

    # 1) Prefer match by advertised service UUID (more robust than name on Windows)
    for addr, (dev, adv) in found.items():
        if _has_uart_service(adv):
            return dev, adv

    # 2) Fallback: match by local name if service UUID isn't present in adv packets
    for addr, (dev, adv) in found.items():
        dev_name = dev.name or getattr(adv, "local_name", None)
        if dev_name == BLE_DEVICE_NAME:
            return dev, adv

    return None, None

async def _ble_emg_loop(emg_buf: EMGBuffer, stop_flag):
    dev, adv = await _ble_find_device(SCAN_TIMEOUT_SEC)
    if dev is None:
        raise RuntimeError(
            f"[EMG][BLE] Cannot find device advertising UART service {UART_SERVICE_UUID} "
            f"or name={BLE_DEVICE_NAME!r} (scan {SCAN_TIMEOUT_SEC}s)"
        )

    print(f"[EMG][BLE] Found {dev.address} name={dev.name} adv_name={getattr(adv, 'local_name', None)}")
    async with BleakClient(dev) as client:
        print(f"[EMG][BLE] Connected={client.is_connected}")

        def on_notify(_, data: bytearray):
            # Notify payload may be fragmented; buffer until newline
            chunk = data.decode("utf-8", errors="ignore")
            on_notify.buf += chunk
            while "\n" in on_notify.buf:
                one, on_notify.buf = on_notify.buf.split("\n", 1)
                item = parse_emg_line(one)
                if item:
                    emg_buf.add(item)

        on_notify.buf = ""
        await client.start_notify(UART_TX_UUID, on_notify)
        print(f"[EMG][BLE] Notifying on {UART_TX_UUID}")
        try:
            while not stop_flag["stop"]:
                await asyncio.sleep(0.05)
        finally:
            try:
                await client.stop_notify(UART_TX_UUID)
            except Exception:
                pass
            print("[EMG][BLE] Closed")

def _emg_reader_loop_ble(emg_buf: EMGBuffer, stop_flag):
    try:
        asyncio.run(_ble_emg_loop(emg_buf, stop_flag))
    except Exception as e:
        print(f"[EMG][BLE] ERROR: {e}")

def _emg_reader_loop_serial(port, baud, emg_buf: EMGBuffer, stop_flag):
    import serial  # lazy import; only needed if EMG_SOURCE="SERIAL"
    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(1.5)
    ser.reset_input_buffer()
    print(f"[EMG][SERIAL] Connected {port} @ {baud}")
    try:
        while not stop_flag["stop"]:
            try:
                line = ser.readline().decode("utf-8", errors="ignore")
            except:
                continue
            item = parse_emg_line(line)
            if item:
                emg_buf.add(item)
    finally:
        ser.close()
        print("[EMG][SERIAL] Closed")

def emg_reader_loop(emg_buf: EMGBuffer, stop_flag):
    """Dispatcher to read EMG lines into EMGBuffer."""
    if EMG_SOURCE.upper() == "BLE":
        return _emg_reader_loop_ble(emg_buf, stop_flag)
    elif EMG_SOURCE.upper() == "SERIAL":
        return _emg_reader_loop_serial(SERIAL_PORT, BAUD, emg_buf, stop_flag)
    else:
        raise ValueError(f"Unknown EMG_SOURCE={EMG_SOURCE!r}; use 'BLE' or 'SERIAL'")

# ======================
# Main (pose loop + integrator)
# ======================
def main():
    pose_model = load_pose_model(POSE_MODEL_PATH)

    # MediaPipe
    mp_pose = mp_solutions.pose
    mp_drawing = mp_solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # EMG thread
    emg_buf = EMGBuffer(keep_sec=EMG_KEEP_SEC)
    stop_flag = {"stop": False}
    th = threading.Thread(target=emg_reader_loop, args=(emg_buf, stop_flag), daemon=True)
    th.start()

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    # event state
    in_event = False
    event_pose = []
    event_t0 = None
    event_t1 = None
    prev_pose_xyz = None
    stop_low_count = 0

    print("[SYS] Running. Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            t_pc = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            # extract 33x3
            if results.pose_landmarks:
                xyz = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                xyz = np.zeros((33,3), dtype=np.float32)

            # motion trigger
            speed = pose_motion_energy(xyz, prev_pose_xyz)
            prev_pose_xyz = xyz

            if not in_event:
                # start condition
                if speed >= START_SPEED_TH:
                    in_event = True
                    event_pose = [xyz]
                    event_t0 = t_pc
                    stop_low_count = 0
                    cv2.putText(frame, "EVENT: START", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                event_pose.append(xyz)
                # stop condition
                if speed < STOP_SPEED_TH:
                    stop_low_count += 1
                else:
                    stop_low_count = 0

                cv2.putText(frame, f"EVENT: REC  len={len(event_pose)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                if stop_low_count >= STOP_HOLD_FRAMES:
                    in_event = False
                    event_t1 = t_pc

                    # --- build pose input ---
                    pose_xyz_seq = np.stack(event_pose, axis=0)               # (T,33,3)
                    pose_2d = preprocess_pose_seq(pose_xyz_seq)              # (T,99)
                    pose_110 = resample_to_fixed_length(pose_2d, POSE_TARGET_FRAMES)  # (110,99)

                    x = torch.from_numpy(pose_110[None, :, :])               # (1,110,99)
                    with torch.no_grad():
                        logits = pose_model(x)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        cls = int(np.argmax(probs))

                    # --- align EMG by event time ---
                    emg_slice = emg_buf.slice_by_time(event_t0, event_t1)
                    if emg_slice:
                        Lp = float(np.mean([e["Lp"] for e in emg_slice]))
                        Rp = float(np.mean([e["Rp"] for e in emg_slice]))
                        mag = float(np.mean([e["magnitude"] for e in emg_slice]))
                        imb = float(np.mean([e["imbalance"] for e in emg_slice]))
                    else:
                        Lp = Rp = mag = imb = float("nan")

                    event = {
                        "t0": event_t0,
                        "t1": event_t1,
                        "duration_sec": float(event_t1 - event_t0),
                        "pose": {
                            "action_class": cls,
                            "probs": [float(p) for p in probs.tolist()],
                            "frames_raw": len(event_pose),
                            "frames_used": POSE_TARGET_FRAMES,
                        },
                        "emg": {
                            "Lp_mean": Lp,
                            "Rp_mean": Rp,
                            "magnitude_mean": mag,
                            "imbalance_mean": imb,
                            "samples": len(emg_slice),
                        }
                    }

                    print("\n=== EVENT ===")
                    print(json.dumps(event, ensure_ascii=False, indent=2))

            # UI overlay
            cv2.putText(frame, f"speed={speed:.4f}  in_event={in_event}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow("Integrator", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ord('Q')):
                break

    finally:
        stop_flag["stop"] = True
        time.sleep(0.2)
        cap.release()
        cv2.destroyAllWindows()
        print("[SYS] exit")

if __name__ == "__main__":
    main()
