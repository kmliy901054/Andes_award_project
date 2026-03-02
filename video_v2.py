import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import asyncio
import threading
from collections import deque

from bleak import BleakScanner, BleakClient

import torch
import torch.nn as nn

from Breeze import BreezeCoach

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QWidget,
    QPushButton, QTextEdit, QFrame, QProgressBar
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal


# ======================
# DEBUG 開關
# ======================
DEBUG_FORCE_LLM = False
DEBUG_LLM_EVERY_EVENT = False
DEBUG_RUN_LLM_ON_START = False
DEBUG_DISABLE_BLE = False

# ===== LLM 控制（關鍵）=====
LLM_DROP_IF_BUSY = False     # 不忙丟掉，改成排隊
LLM_MAX_QUEUE = 20           # 至少要 > 1，不然只留最新還是會掉
COACH_COOLDOWN_SEC = 0.0     # 不要節流
LLM_EXPIRE_SEC = 4.0         # 超過就丟掉（避免慢半拍）

# ✅ 新增：語音/回應一致性驗證參數
LLM_REP_DRIFT_MAX = 2        # LLM 回來時，rep 差 >= 2 就丟掉（避免講第21下但你已經第27下）
LLM_REQUIRE_SIDE_MATCH = True  # side_state 不同就丟掉（避免左/右講反）

# ===== EMG 左右穩定判斷 =====
EMG_IMB_ALPHA = 0.2
EMG_IMB_HI = 0.08
EMG_IMB_LO = 0.04
EMG_SIDE_STABLE_SEC = 0.6


# ======================
# BLE (NUS) UUIDs
# ======================
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_UUID      = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_DEVICE_NAME_FALLBACK = "CorvetteT1-EMG"

def has_uart_service(adv):
    uuids = getattr(adv, "service_uuids", None) or []
    uuids = [u.lower() for u in uuids]
    return UART_SERVICE_UUID.lower() in uuids


# ======================
# Pose LSTM 設定（配合你訓練）
# ======================
POSE_MODEL_PATH = "pose_lstm.pt"
NUM_CLASSES = 4
POSE_DIM = 99
POSE_TARGET_FRAMES = 110

LABELS = {
    0: "正常",
    1: "左邊太高",
    2: "右邊太高",
    3: "兩邊都太高",
}

# ===== Event 偵測參數（保留，用於「一段動作結束」才跑 PoseLSTM）=====
SPEED_JOINTS = [15, 16, 11, 12, 23, 24]
START_SPEED_TH = 0.010
STOP_SPEED_TH  = 0.006
STOP_HOLD_FRAMES = 12

# Idle / Trigger
EMG_KEEP_SEC = 15.0
IDLE_MAG_TH = 0.08
COACH_COOLDOWN_SEC = 2.0   # LLM 類語音（慢語音）冷卻

# ===== 統一定義 =====
MIN_EVENT_DURATION_SEC = 1.2
MAX_PROMPT_CHARS = 300

# ===== rep 偵測（不用停很久也能分 rep）=====
REP_MIN_RANGE = 0.06
REP_REFRACT_SEC = 0.40
REP_SMOOTH_ALPHA = 0.35
REP_LOW_FRAC = 0.70
REP_HIGH_FRAC = 0.35
ENCOURAGE_EVERY_N_REPS = 2     # 每 3 下用 LLM 鼓勵一次


# ======================
# EMG 解析 + buffer
# ======================
def parse_emg_csv_line(line: str):
    line = line.strip()
    if not line:
        return None
    parts = line.split(",")
    tmillis = None

    if len(parts) == 7:
        try:
            tmillis = float(parts[0])
        except:
            tmillis = None
        parts = parts[1:]

    if len(parts) != 6:
        return None

    try:
        Lp, Rp, imbalance, magnitude, finalL, finalR = map(float, parts)
        out = {
            "t_pc": time.time(),
            "Lp": Lp,
            "Rp": Rp,
            "imbalance": imbalance,
            "magnitude": magnitude,
            "finalL": finalL,
            "finalR": finalR,
        }
        if tmillis is not None:
            out["tmillis"] = tmillis
        return out
    except:
        return None


class EMGBuffer:
    def __init__(self, keep_sec=15.0):
        self.keep_sec = keep_sec
        self.buf = deque()
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


# ======================
# Pose preprocessing
# ======================
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24

def preprocess_pose_seq(pose_xyz_seq: np.ndarray) -> np.ndarray:
    T = pose_xyz_seq.shape[0]
    pose = pose_xyz_seq.copy()

    missing = np.isclose(pose, 0.0).all(axis=(1, 2))
    for t in range(1, T):
        if missing[t]:
            pose[t] = pose[t - 1]

    centers = (pose[:, L_HIP] + pose[:, R_HIP]) / 2.0
    scales = np.linalg.norm(pose[:, L_SHO] - pose[:, R_SHO], axis=1)
    scales[scales < 1e-6] = 1.0

    pose = (pose - centers[:, None, :]) / scales[:, None, None]
    return pose.reshape(T, 99).astype(np.float32)

def resample_to_fixed_length(seq_2d: np.ndarray, target_len: int) -> np.ndarray:
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
# Pose model
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
# 角度（只拿來顯示）
# ======================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# ======================
# TTS thread
# ======================
class SpeakerThread(QThread):
    finished_signal = Signal()
    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 250)
            voices = engine.getProperty('voices')
            for v in voices:
                if "Chinese" in v.name or "Han" in v.name:
                    engine.setProperty('voice', v.id)
                    break
            engine.say(self.text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS] ERROR: {e}")
        finally:
            self.finished_signal.emit()


# ======================
# BLE EMG thread
# ======================
class BLEEMGThread(QThread):
    emg_signal = Signal(dict)
    status_signal = Signal(str)

    def __init__(self, scan_timeout=8.0):
        super().__init__()
        self.scan_timeout = scan_timeout
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.status_signal.emit(f"[BLE] ERROR: {e}")

    async def _main(self):
        self.status_signal.emit("[BLE] scanning...")
        found = await BleakScanner.discover(timeout=self.scan_timeout, return_adv=True)

        target_dev = None
        target_adv = None

        for addr, (dev, adv) in found.items():
            if has_uart_service(adv):
                target_dev = dev
                target_adv = adv
                break

        if target_dev is None:
            for addr, (dev, adv) in found.items():
                name = dev.name or getattr(adv, "local_name", None)
                if name == BLE_DEVICE_NAME_FALLBACK:
                    target_dev = dev
                    target_adv = adv
                    break

        if target_dev is None:
            raise RuntimeError("找不到含 NUS service 的裝置（也找不到 CorvetteT1-EMG）。")

        self.status_signal.emit(
            f"[BLE] connect {target_dev.address} name={target_dev.name or getattr(target_adv,'local_name',None)}"
        )

        async with BleakClient(target_dev) as client:
            self.status_signal.emit(f"[BLE] connected={client.is_connected}")

            buf = ""

            def on_notify(_, data: bytearray):
                nonlocal buf
                buf += data.decode("utf-8", errors="ignore")
                while "\n" in buf:
                    one, buf = buf.split("\n", 1)
                    item = parse_emg_csv_line(one)
                    if item:
                        self.emg_signal.emit(item)

            await client.start_notify(UART_TX_UUID, on_notify)
            self.status_signal.emit("[BLE] notifying...")

            while not self._stop and not self.isInterruptionRequested():
                await asyncio.sleep(0.05)

            try:
                await client.stop_notify(UART_TX_UUID)
            except:
                pass
            self.status_signal.emit("[BLE] stopped")


# ======================
# Video thread：event + rep 偵測
# ======================
class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    status_signal = Signal(str)
    event_signal = Signal(dict)
    rep_signal = Signal(int)

    def __init__(self, camera_index=1):
        super().__init__()
        self.camera_index = camera_index

    def run(self):
        try:
            pose_model = load_pose_model(POSE_MODEL_PATH)
        except Exception as e:
            self.status_signal.emit(f"[POSE] load model failed: {e}")
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status_signal.emit("[CAM] open failed")
            return

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        mp_drawing = mp.solutions.drawing_utils

        in_event = False
        event_pose = []
        event_t0 = None
        prev_pose_xyz = None
        stop_low_count = 0

        rep_count = 0
        rep_phase = "UP"
        last_rep_time = 0.0

        delta_smooth = None
        dyn_min = None
        dyn_max = None
        last_seen_pose_time = 0.0

        self.status_signal.emit("IDLE")

        while not self.isInterruptionRequested():
            ret, frame = cap.read()
            if not ret:
                continue

            t_pc = time.time()
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb_img)

            has_pose = bool(res.pose_landmarks)

            if has_pose:
                last_seen_pose_time = t_pc
                lm = res.pose_landmarks.landmark
                xyz = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # rep 指標： (wrist_y_avg - shoulder_y_avg)
                wrist_y = 0.5 * (lm[15].y + lm[16].y)
                sho_y = 0.5 * (lm[11].y + lm[12].y)
                delta = float(wrist_y - sho_y)

                if delta_smooth is None:
                    delta_smooth = delta
                else:
                    delta_smooth = (1.0 - REP_SMOOTH_ALPHA) * delta_smooth + REP_SMOOTH_ALPHA * delta

                if dyn_min is None:
                    dyn_min = delta_smooth
                    dyn_max = delta_smooth
                else:
                    dyn_min = min(dyn_min * 0.995 + delta_smooth * 0.005, delta_smooth)
                    dyn_max = max(dyn_max * 0.995 + delta_smooth * 0.005, delta_smooth)

                rng = float(dyn_max - dyn_min)

                if rng >= REP_MIN_RANGE:
                    high_th = dyn_min + REP_HIGH_FRAC * rng
                    low_th  = dyn_min + REP_LOW_FRAC  * rng

                    if rep_phase == "UP":
                        if delta_smooth >= low_th:
                            rep_phase = "DOWN"
                    else:
                        if delta_smooth <= high_th and (t_pc - last_rep_time) >= REP_REFRACT_SEC:
                            rep_phase = "UP"
                            last_rep_time = t_pc
                            rep_count += 1
                            self.rep_signal.emit(rep_count)

            else:
                xyz = np.zeros((33, 3), dtype=np.float32)
                if (t_pc - last_seen_pose_time) > 2.0:
                    delta_smooth = None
                    dyn_min = None
                    dyn_max = None
                    rep_phase = "UP"

            # event 偵測（段落）
            speed = pose_motion_energy(xyz, prev_pose_xyz)
            prev_pose_xyz = xyz

            if not in_event:
                if has_pose and speed >= START_SPEED_TH:
                    in_event = True
                    event_pose = [xyz]
                    event_t0 = t_pc
                    stop_low_count = 0
                    self.status_signal.emit("RECORDING")
            else:
                event_pose.append(xyz)
                if speed < STOP_SPEED_TH:
                    stop_low_count += 1
                else:
                    stop_low_count = 0

                if stop_low_count >= STOP_HOLD_FRAMES:
                    in_event = False
                    event_t1 = t_pc
                    self.status_signal.emit("END")

                    dur = float(event_t1 - event_t0)
                    if dur >= MIN_EVENT_DURATION_SEC:
                        try:
                            pose_xyz_seq = np.stack(event_pose, axis=0)
                            pose_2d = preprocess_pose_seq(pose_xyz_seq)
                            pose_110 = resample_to_fixed_length(pose_2d, POSE_TARGET_FRAMES)

                            x = torch.from_numpy(pose_110[None, :, :])
                            with torch.no_grad():
                                logits = pose_model(x)
                                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                                cls = int(np.argmax(probs))
                        except Exception as e:
                            self.status_signal.emit(f"[POSE] infer failed: {e}")
                            cls = 0
                            probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

                        self.event_signal.emit({
                            "t0": float(event_t0),
                            "t1": float(event_t1),
                            "duration_sec": float(dur),
                            "frames_raw": int(len(event_pose)),
                            "frames_used": int(POSE_TARGET_FRAMES),
                            "cls": int(cls),
                            "probs": [float(p) for p in probs.tolist()],
                        })

                    self.status_signal.emit("IDLE")

            self.change_pixmap_signal.emit(frame.copy())

        cap.release()


# ======================
# LLM worker（單一執行緒 + queue）
# ======================
class LLMWorkerThread(QThread):
    result_signal = Signal(str, dict)
    status_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self._coach = BreezeCoach(enable_warmup=False)
        self._queue = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._busy = False

    def stop(self):
        with self._lock:
            self._stop = True
            self._cv.notify_all()

    def enqueue(self, exercise, geometry_status, lstm_status, meta: dict):
        with self._lock:
            if LLM_DROP_IF_BUSY and self._busy:
                return False

            self._queue.append((exercise, geometry_status, lstm_status, meta))

            # 只保留最新
            while len(self._queue) > LLM_MAX_QUEUE:
                self._queue.popleft()

            self._cv.notify()
            return True

    def run(self):
        while True:
            with self._lock:
                while not self._queue and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                exercise, geometry_status, lstm_status, meta = self._queue.popleft()
                self._busy = True

            # ✅ 如果排隊等太久，直接不算（減少無謂的 LLM 計算）
            if time.time() - meta.get("created", 0) > LLM_EXPIRE_SEC:
                with self._lock:
                    self._busy = False
                self.status_signal.emit("LLM: idle")
                continue

            try:
                self.status_signal.emit("LLM: busy")
                t0 = time.time()
                text = self._coach.get_advice(exercise, geometry_status, lstm_status)
                meta["dt"] = time.time() - t0
                self.result_signal.emit(text, meta)
            except Exception:
                self.result_signal.emit("教練暫時無法回應", meta)
            finally:
                with self._lock:
                    self._busy = False
                self.status_signal.emit("LLM: idle")


# ======================
# UI
# ======================
class FitnessUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.latest_llm_request_id = None

        self.setWindowTitle("智慧健身矯正系統 - Event + RepCounter + PoseLSTM + BLE EMG + Ollama")
        self.resize(1200, 980)
        self.setStyleSheet("QMainWindow { background-color: #121212; }")

        self.setup_ui()

        # ✅ EMG 左右穩定器狀態（放在 __init__ 最合理）
        self.imb_smooth = 0.0
        self.side_state = "平衡"
        self.side_candidate = "平衡"
        self.side_candidate_t0 = time.time()

        self.emg_buf = EMGBuffer(keep_sec=EMG_KEEP_SEC)

        # rep 計數
        self.rep_count = 0
        self.last_encourage_rep = 0

        # 快語音節流
        self.last_fast_voice_time = 0.0

        # LLM worker
        self.llm_worker = LLMWorkerThread()
        self.llm_worker.result_signal.connect(self.on_llm_result)
        self.llm_worker.status_signal.connect(self.on_llm_status)
        self.llm_worker.start()

        # BLE thread
        self.ble_thread = None
        if not DEBUG_DISABLE_BLE:
            self.ble_thread = BLEEMGThread(scan_timeout=8.0)
            self.ble_thread.emg_signal.connect(self.on_emg_update)
            self.ble_thread.status_signal.connect(self.on_ble_status)
            self.ble_thread.start()

        # video thread
        self.v_thread = VideoThread(camera_index=1)
        self.v_thread.change_pixmap_signal.connect(self.update_img)
        self.v_thread.status_signal.connect(self.on_video_status)
        self.v_thread.event_signal.connect(self.on_event)
        self.v_thread.rep_signal.connect(self.on_rep)

        # voice
        self.speech_threads = []
        self.pending_tts = []
        self.is_speaking = False

        self.last_voice_time = 0.0  # LLM 類語音節流

        # 最近姿勢
        self.last_pose_cls = 0
        self.last_pose_label = "正常"
        self.last_pose_probs = [1.0, 0.0, 0.0, 0.0]

    def setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # left
        self.left_col = QVBoxLayout()
        self.image_label = QLabel("鏡頭影像區域")
        self.image_label.setFixedSize(800, 580)
        self.image_label.setStyleSheet("border: 3px solid #333; background: black; border-radius: 15px;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.left_col.addWidget(self.image_label)

        self.btn_start = QPushButton("▶ 啟動 AI 健身教練模式")
        self.btn_start.setFixedHeight(70)
        self.btn_start.setStyleSheet(
            "background-color: #0078D4; color: white; font-size: 22px; font-weight: bold; "
            "border-radius: 12px; margin-top: 10px;"
        )
        self.btn_start.clicked.connect(self.start_system)
        self.left_col.addWidget(self.btn_start)

        self.main_layout.addLayout(self.left_col, stretch=3)

        # right
        self.right_col = QVBoxLayout()

        # class / rep
        self.card_cls = QFrame()
        self.card_cls.setStyleSheet("background: #1E1E1E; border-radius: 15px; padding: 10px;")
        lc = QVBoxLayout(self.card_cls)
        lc.addWidget(QLabel("PoseLSTM 類別 & Rep", styleSheet="color: #888; font-size: 14px;"))
        self.lbl_cls = QLabel("--")
        self.lbl_cls.setStyleSheet("color: #7FDBFF; font-size: 24px; font-weight: bold;")
        lc.addWidget(self.lbl_cls)
        self.right_col.addWidget(self.card_cls)

        # state
        self.card_state = QFrame()
        self.card_state.setStyleSheet("background: #1E1E1E; border-radius: 15px; padding: 10px;")
        ls = QVBoxLayout(self.card_state)
        ls.addWidget(QLabel("狀態", styleSheet="color: #888; font-size: 14px;"))
        self.lbl_state = QLabel("IDLE")
        self.lbl_state.setStyleSheet("color: #AAAAAA; font-size: 26px; font-weight: bold;")
        ls.addWidget(self.lbl_state)
        self.right_col.addWidget(self.card_state)

        # LLM status
        self.card_llm = QFrame()
        self.card_llm.setStyleSheet("background: #1E1E1E; border-radius: 15px; padding: 10px;")
        ll = QVBoxLayout(self.card_llm)
        ll.addWidget(QLabel("LLM 狀態", styleSheet="color: #888; font-size: 14px;"))
        self.lbl_llm = QLabel("LLM: idle")
        self.lbl_llm.setStyleSheet("color: #AAAAAA; font-size: 18px; font-weight: bold;")
        ll.addWidget(self.lbl_llm)
        self.right_col.addWidget(self.card_llm)

        # score bar
        # self.card_score = QFrame()
        # self.card_score.setStyleSheet("background: #1E1E1E; border-radius: 15px; padding: 15px;")
        # lsc = QVBoxLayout(self.card_score)
        # lsc.addWidget(QLabel("信心分數(0~100)", styleSheet="color: #888; font-size: 14px;"))
        # self.score_bar = QProgressBar()
        # self.score_bar.setRange(0, 100)
        # self.score_bar.setValue(0)
        # self.score_bar.setFixedHeight(30)
        # self.score_bar.setTextVisible(True)
        # self.score_bar.setStyleSheet("""
        #     QProgressBar { border: 1px solid #444; border-radius: 5px; background: #222; text-align: center; color: white; font-weight: bold; }
        #     QProgressBar::chunk { background-color: #00FF00; }
        # """)
        # lsc.addWidget(self.score_bar)
        # self.right_col.addWidget(self.card_score)

        # EMG card
        self.card_emg = QFrame()
        self.card_emg.setStyleSheet("background: #1E1E1E; border-radius: 15px; padding: 10px;")
        le = QVBoxLayout(self.card_emg)
        le.addWidget(QLabel("EMG 即時狀態", styleSheet="color: #888; font-size: 14px;"))
        self.lbl_emg = QLabel("等待 EMG...")
        self.lbl_emg.setStyleSheet("color: #FFD700; font-size: 18px; font-weight: bold;")
        le.addWidget(self.lbl_emg)
        self.right_col.addWidget(self.card_emg)

        # coach box
        self.coach_box = QTextEdit()
        self.coach_box.setReadOnly(True)
        self.coach_box.setStyleSheet("background: #252525; color: #EEE; font-size: 15px; border-radius: 10px; padding: 10px;")
        self.right_col.addWidget(self.coach_box)

        self.main_layout.addLayout(self.right_col, stretch=1)

    def start_system(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setText("🔍 系統偵測分析中...")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2D2D2D;
                color: #00FF00;
                font-size: 22px;
                font-weight: bold;
                border-radius: 12px;
                margin-top: 10px;
                border: 2px solid #00FF00;
            }
        """)
        self.v_thread.start()
        self.send_coach_msg("你好，我是你的虛擬健身教練")

        if DEBUG_RUN_LLM_ON_START:
            req_id = f"start_{int(time.time()*1000)}"
            self.latest_llm_request_id = req_id
            self.llm_worker.enqueue(
                "滑輪下拉",
                "DEBUG start",
                "DEBUG no emg",
                meta={
                    "request_id": req_id,
                    "created": time.time(),
                    "type": "debug",
                    "rep": self.rep_count,
                    "side_state": self.side_state,
                    "pose_cls": self.last_pose_cls,
                    "pose_label": self.last_pose_label,
                }
            )

    # ---------- UI update ----------
    def update_img(self, frame):
        h, w, ch = frame.shape
        img = QImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).data, w, h, ch*w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(img).scaled(800, 580, Qt.KeepAspectRatio))

    def on_video_status(self, s: str):
        if s == "IDLE":
            self.lbl_state.setText("IDLE")
            self.lbl_state.setStyleSheet("color: #AAAAAA; font-size: 26px; font-weight: bold;")
        elif s == "RECORDING":
            self.lbl_state.setText("RECORDING")
            self.lbl_state.setStyleSheet("color: #FFA500; font-size: 26px; font-weight: bold;")
        else:
            self.lbl_state.setText(s)
            self.lbl_state.setStyleSheet("color: #7FDBFF; font-size: 26px; font-weight: bold;")

    def on_llm_status(self, s: str):
        self.lbl_llm.setText(s)
        if "busy" in s:
            self.lbl_llm.setStyleSheet("color: #FFA500; font-size: 18px; font-weight: bold;")
        else:
            self.lbl_llm.setStyleSheet("color: #AAAAAA; font-size: 18px; font-weight: bold;")

    def on_ble_status(self, s: str):
        print(s)

    # ✅ EMG：即時更新 buffer + UI（含左右穩定器）
    def on_emg_update(self, emg: dict):
        self.emg_buf.add(emg)

        mag = float(emg.get("magnitude", 0.0))
        imb = float(emg.get("imbalance", 0.0))
        Lp = float(emg.get("Lp", 0.0))
        Rp = float(emg.get("Rp", 0.0))

        # EMA 平滑
        self.imb_smooth = (1 - EMG_IMB_ALPHA) * self.imb_smooth + EMG_IMB_ALPHA * imb

        # hysteresis + stable hold
        desired = self.side_state
        if self.side_state == "平衡":
            if self.imb_smooth > EMG_IMB_HI:
                desired = "左側偏多"
            elif self.imb_smooth < -EMG_IMB_HI:
                desired = "右側偏多"
        elif self.side_state == "左側偏多" and self.imb_smooth < EMG_IMB_LO:
            desired = "平衡"
        elif self.side_state == "右側偏多" and self.imb_smooth > -EMG_IMB_LO:
            desired = "平衡"

        now = time.time()
        if desired != self.side_candidate:
            self.side_candidate = desired
            self.side_candidate_t0 = now
        elif now - self.side_candidate_t0 >= EMG_SIDE_STABLE_SEC:
            self.side_state = desired

        self.lbl_emg.setText(
            f"力量: {mag:.2f}\n"
            f"左右: {self.side_state} (imb~={self.imb_smooth:.2f})\n"
            f"L: {Lp:.2f} | R: {Rp:.2f}"
        )

    def update_score_ui(self, score01: float):
        percentage = int(max(0.0, min(1.0, score01)) * 100)
        self.score_bar.setValue(percentage)
        color = "#00FF00" if percentage >= 80 else "#FFA500" if percentage >= 60 else "#FF0000"
        self.score_bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid #444; border-radius: 5px; background: #222; text-align: center; color: white; }}
            QProgressBar::chunk {{ background-color: {color}; }}
        """)

    # ✅ rep：即時 + 鼓勵
    def on_rep(self, rep_count: int):
        self.rep_count = int(rep_count)

        # UI 顯示
        self.lbl_cls.setText(f"Rep: {self.rep_count} | Pose: {self.last_pose_cls} {self.last_pose_label}")

        # ===== 每 3 下：用 LLM 給一句比較像教練的鼓勵 =====
        if self.rep_count > 0:
            self.last_encourage_rep = self.rep_count

            now = time.time()
            self.last_voice_time = now

            # ✅ prompt 要用「快照」描述（避免它用現在式講過去）
            geometry_status = (
                f"使用者剛完成第{self.rep_count}下，請給一句鼓勵（20~35字）。"
                f"剛才姿勢判斷={self.last_pose_cls}({self.last_pose_label})。"
                f"EMG左右狀態={self.side_state}。"
            )[:MAX_PROMPT_CHARS]

            lstm_status = self.lbl_emg.text().replace("\n", "，")[:MAX_PROMPT_CHARS]

            req_id = f"rep_{self.rep_count}_{int(time.time()*1000)}"
            self.latest_llm_request_id = req_id

            self.llm_worker.enqueue(
                "滑輪下拉",
                geometry_status,
                lstm_status,
                meta={
                    "request_id": req_id,
                    "created": time.time(),
                    "type": "encourage",

                    # ===== 狀態快照（超重要）=====
                    "rep": self.rep_count,
                    "pose_cls": self.last_pose_cls,
                    "pose_label": self.last_pose_label,
                    "side_state": self.side_state,
                }
            )

    # ✅ event 結束：PoseLSTM +（必要時）矯正
    def on_event(self, ev: dict):
        t0 = float(ev["t0"])
        t1 = float(ev["t1"])
        cls = int(ev["cls"])
        probs = np.array(ev["probs"], dtype=np.float32)
        label = LABELS.get(cls, f"未知{cls}")

        self.last_pose_cls = cls
        self.last_pose_label = label
        self.last_pose_probs = probs.tolist()

        # EMG slice
        emg_slice = self.emg_buf.slice_by_time(t0, t1)
        if emg_slice:
            Lp = float(np.mean([e["Lp"] for e in emg_slice]))
            Rp = float(np.mean([e["Rp"] for e in emg_slice]))
            mag = float(np.mean([e["magnitude"] for e in emg_slice]))
            imb = float(np.mean([e["imbalance"] for e in emg_slice]))
        else:
            Lp = Rp = mag = imb = float("nan")

        p0 = float(probs[0]) if probs.size >= 1 else 0.0
        self.update_score_ui(p0)

        # UI 顯示
        self.lbl_cls.setText(f"Rep: {self.rep_count} | Pose: {cls} {label} (P0={p0:.2f})")

        should_coach = DEBUG_LLM_EVERY_EVENT or (cls != 0)

        print(
            f"[EVENT] cls={cls}({label}) P0={p0:.2f} "
            f"mag={mag:.3f} imb={imb:.3f} samples={len(emg_slice)} dur={ev['duration_sec']:.2f}s"
        )

        now = time.time()
        if should_coach and (now - self.last_voice_time > COACH_COOLDOWN_SEC):
            self.last_voice_time = now

            geometry_status = (
                f"以下是『剛結束的一段動作』分析，請給1句矯正建議（20~40字）。"
                f"PoseLSTM結果：cls={cls}({label}), probs={np.round(probs, 3).tolist()}, "
                f"dur={ev['duration_sec']:.2f}s"
            )[:MAX_PROMPT_CHARS]

            lstm_status = (
                f"EMG段落平均：mag_mean={mag:.3f}, imb_mean(Lp-Rp)={imb:.3f}, "
                f"Lp_mean={Lp:.3f}, Rp_mean={Rp:.3f}, samples={len(emg_slice)}"
            )[:MAX_PROMPT_CHARS]

            req_id = f"event_{int(t1*1000)}_{int(time.time()*1000)}"
            self.latest_llm_request_id = req_id

            self.llm_worker.enqueue(
                "滑輪下拉",
                geometry_status,
                lstm_status,
                meta={
                    "request_id": req_id,
                    "created": time.time(),
                    "type": "correct",

                    # ===== 狀態快照 =====
                    "rep": self.rep_count,
                    "pose_cls": cls,
                    "pose_label": label,
                    "side_state": self.side_state,
                }
            )

    # ✅ LLM 回來：做「一致性檢查」，不一致就不講
    def on_llm_result(self, txt: str, meta: dict):
        # ❷ 時間過期
        if time.time() - meta.get("created", 0) > LLM_EXPIRE_SEC:
            return

        # ❸ rep 漂移太多（避免講第21下但你已經第27下）
        rep_snap = meta.get("rep", None)
        if rep_snap is not None:
            if abs(self.rep_count - int(rep_snap)) >= LLM_REP_DRIFT_MAX:
                return

        # ❹ 左右狀態不一致（避免講反）
        if LLM_REQUIRE_SIDE_MATCH:
            if meta.get("side_state") is not None and meta.get("side_state") != self.side_state:
                return

        self.send_coach_msg(txt)

    # ---------- TTS queue ----------
    def start_tts(self, txt):
        self.is_speaking = True
        speaker = SpeakerThread(txt)
        speaker.finished_signal.connect(self.on_speech_finished)
        self.speech_threads.append(speaker)
        speaker.start()

    def send_coach_msg(self, txt):
        self.coach_box.append(f"<b style='color:#0078D4;'>教練：</b>{txt}")
        self.pending_tts.append(txt)

        # 防止堆太多（避免跟不上）
        if len(self.pending_tts) > 5:
            self.pending_tts = self.pending_tts[-5:]

        if not self.is_speaking:
            nxt = self.pending_tts.pop(0)
            self.start_tts(nxt)

    def on_speech_finished(self):
        self.is_speaking = False
        self.speech_threads = [t for t in self.speech_threads if t.isRunning()]
        if self.pending_tts:
            nxt = self.pending_tts.pop(0)
            self.start_tts(nxt)

    def closeEvent(self, event):
        self.v_thread.requestInterruption()
        self.v_thread.wait()

        if self.ble_thread is not None and self.ble_thread.isRunning():
            self.ble_thread.stop()
            self.ble_thread.requestInterruption()
            self.ble_thread.wait(1000)

        if hasattr(self, "llm_worker") and self.llm_worker.isRunning():
            self.llm_worker.stop()
            self.llm_worker.requestInterruption()
            self.llm_worker.wait(1500)

        for t in self.speech_threads:
            t.wait(200)

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FitnessUI()
    win.show()
    sys.exit(app.exec())
