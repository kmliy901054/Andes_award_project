import cv2
import serial
import time
import torch
import numpy as np
import torch.nn.functional as F
import csv
import os

# ================= 參數設定 =================
SERIAL_PORT = "COM9"        # 請確認 Corvette T1 的 Port
BAUD_RATE = 115200
CAMERA_INDEX = 1            # 0: 筆電, 1: 外接
ACTION_DURATION = 5.0       # 動作時間（秒）
TARGET_SAMPLES = 110        # Arduino 固定輸出 110 筆
EMG_DIM = 2                 # finalL, finalR
POSE_DIM = 99               # 33 landmarks * (x,y,z) = 99
OUTPUT_FOLDER = "collected_data"

# 讀 Arduino 的等待時間（秒）
SERIAL_LINE_TIMEOUT = 0.05  # 建議縮短，讀取更即時
POST_READ_SECONDS = 3.0     # Pose錄完後，再補讀最多幾秒（通常足夠把110行吃完）
MIN_VALID_ROWS = 100        # 少於這個視為失敗（你可改成110更嚴格）

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= MediaPipe 初始化 =================
try:
    from mediapipe import solutions as mp_solutions
except ImportError as e:
    raise ImportError(
        "Cannot import mediapipe.solutions. Please reinstall/upgrade mediapipe.\n"
        "Try: pip uninstall mediapipe -y && pip install mediapipe==0.10.9"
    ) from e

mp_pose = mp_solutions.pose
mp_drawing = mp_solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ================= Serial 初始化 =================
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_LINE_TIMEOUT)
    time.sleep(2)
    ser.reset_input_buffer()
    print(f"[OK] Connected to Corvette T1 at {SERIAL_PORT}")
except Exception as e:
    print(f"[Serial Error] {e}")
    raise SystemExit(1)

# ================= Camera 初始化 =================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[Warn] Camera {CAMERA_INDEX} failed. Trying 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Camera Error] No camera available.")
        raise SystemExit(1)

cap.set(cv2.CAP_PROP_FPS, 30)

# ================= 核心函式 =================
def extract_pose_landmarks(results):
    """回傳 99 維 (33 * xyz)，沒有偵測到就回全 0。"""
    if results.pose_landmarks:
        data = []
        for lm in results.pose_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return data
    return [0.0] * POSE_DIM


def try_parse_emg_line(line: str):
    """
    單行解析：格式 "0.123456,0.654321"
    成功 -> 回傳 [L,R] (float)
    失敗 -> 回傳 None
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    if len(parts) != EMG_DIM:
        return None
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError:
        return None


def drain_emg_available():
    """
    非阻塞：把目前 serial buffer 裡「能讀到的」行都讀出來
    回傳: raw_lines(list[str]), parsed_vals(list[list[float]])
    """
    raw = []
    parsed = []
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not line:
            continue
        raw.append(line)
        vals = try_parse_emg_line(line)
        if vals is not None:
            parsed.append(vals)
    return raw, parsed


def align_emg_to_target(emg_raw: np.ndarray, target_len=TARGET_SAMPLES):
    """
    將有效 EMG 筆數 N 對齊到 target_len：
    - N < target_len: 用最後一筆 padding
    - N > target_len: 截斷
    回傳 shape=(target_len,2)
    """
    if emg_raw.size == 0:
        return np.zeros((0, EMG_DIM), dtype=np.float32)

    emg = emg_raw
    if emg.shape[0] < target_len:
        pad = np.tile(emg[-1], (target_len - emg.shape[0], 1))
        emg = np.vstack([emg, pad])
    elif emg.shape[0] > target_len:
        emg = emg[:target_len]
    return emg.astype(np.float32)


def resample_pose_data(raw_pose_data, target_len=TARGET_SAMPLES):
    """將 pose 序列重採樣到 target_len，輸出 (110,99)"""
    if len(raw_pose_data) == 0:
        return np.zeros((target_len, POSE_DIM), dtype=np.float32)

    data_tensor = torch.tensor(raw_pose_data, dtype=torch.float32)  # (T,99)
    data_tensor = data_tensor.permute(1, 0).unsqueeze(0)            # (1,99,T)

    resized = F.interpolate(
        data_tensor, size=target_len, mode="linear", align_corners=False
    )                                                               # (1,99,110)

    out = resized.squeeze(0).permute(1, 0).cpu().numpy()            # (110,99)
    return out.astype(np.float32)


def save_to_csv(data, filename):
    full_path = os.path.join(OUTPUT_FOLDER, filename)

    header = ["EMG_L_Norm", "EMG_R_Norm"]
    for i in range(33):
        header.extend([f"Node{i}_X", f"Node{i}_Y", f"Node{i}_Z"])

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"[SAVED] {full_path}")


# ================= 採集邏輯 =================
def collect_one_sample(sample_index):
    filename = f"action_{sample_index:03d}.csv"
    print("-------------------------------------------------")
    print(f"Ready for Sample #{sample_index:03d} -> {filename}")

    # 1) 預覽：Enter 開始、ESC 離開
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera Error] Failed to read frame.")
            return False, False

        cv2.putText(frame, f"Next: {filename}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Press ENTER to Record | ESC to Exit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1)

        if key == 13:      # Enter
            break
        elif key == 27:    # ESC
            return False, False

    # 2) 錄製 5 秒 Pose + 觸發 Arduino
    print(f">>> Recording Sample #{sample_index:03d} for {ACTION_DURATION:.1f}s")

    # 清掉上一輪殘留（一定要在送 S 前）
    ser.reset_input_buffer()

    # 觸發 Arduino（你的 Arduino 是收到 S 才會送 5 秒 / 110 行）
    ser.write(b"S\n")

    start_time = time.time()
    pose_buffer = []

    # ✅ 同步收 EMG（邊錄 Pose 邊吃 serial）
    arduino_raw_lines = []
    emg_vals = []

    while True:
        elapsed = time.time() - start_time
        if elapsed >= ACTION_DURATION:
            break

        # 先把目前 serial 裡的資料吃掉（非阻塞）
        raw_chunk, parsed_chunk = drain_emg_available()
        if raw_chunk:
            arduino_raw_lines.extend(raw_chunk)
        if parsed_chunk:
            emg_vals.extend(parsed_chunk)

        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        pose_vector = extract_pose_landmarks(results)
        pose_buffer.append(pose_vector)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        cv2.putText(frame, f"REC: {elapsed:.2f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)

    # 3) 錄完後補讀剩下的 EMG（最多等 POST_READ_SECONDS）
    print(">>> Waiting for Corvette T1 data (post-drain)...")
    deadline = time.time() + POST_READ_SECONDS
    while len(emg_vals) < TARGET_SAMPLES and time.time() < deadline:
        raw_chunk, parsed_chunk = drain_emg_available()
        if raw_chunk:
            arduino_raw_lines.extend(raw_chunk)
        if parsed_chunk:
            emg_vals.extend(parsed_chunk)
        time.sleep(0.01)

    emg_raw = np.array(emg_vals, dtype=np.float32) if len(emg_vals) > 0 else np.zeros((0, EMG_DIM), dtype=np.float32)
    raw_valid = emg_raw.shape[0]

    print(f"[INFO] Arduino raw lines: {len(arduino_raw_lines)}, valid EMG rows(raw): {raw_valid}")

    if raw_valid < MIN_VALID_ROWS:
        print(f"[Error] EMG data insufficient (Got {raw_valid} valid rows).")

        # 把 raw 存下來方便 debug（最多存前 500 行）
        fail_path = os.path.join(OUTPUT_FOLDER, f"fail_{sample_index:03d}_raw.txt")
        try:
            with open(fail_path, "w", encoding="utf-8") as f:
                f.write("\n".join(arduino_raw_lines[:500]))
            print(f"[DEBUG] Saved raw lines to: {fail_path}")
        except Exception as e:
            print(f"[DEBUG] Failed to save raw debug file: {e}")

        return True, False

    # 對齊 110
    emg_data = align_emg_to_target(emg_raw, target_len=TARGET_SAMPLES)  # (110,2)

    # Pose 重採樣到 110
    pose_data = resample_pose_data(pose_buffer, target_len=TARGET_SAMPLES)  # (110,99)

    # 合併輸出
    final_dataset = np.hstack([emg_data, pose_data])  # (110,101)
    save_to_csv(final_dataset, filename)

    return True, True


# ================= 主程式 =================
try:
    current_idx = 1
    while True:
        cont, success = collect_one_sample(current_idx)
        if not cont:
            break
        if success:
            current_idx += 1
finally:
    cap.release()
    ser.close()
    cv2.destroyAllWindows()
    print("[DONE] Closed camera, serial, and windows.")
