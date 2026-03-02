import cv2
import mediapipe as mp
import numpy as np
import os

# --- 初始化 MediaPipe ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- 設定資料夾 ---
input_dir = 'Training_Data'
output_dir = "pose_output"
os.makedirs(output_dir, exist_ok=True)

exit_flag = False # <--- 優化1: 用於完全退出的旗標

# --- 遍歷影片檔案 ---
for file in os.listdir(input_dir):
    # <--- 修正2: 使用 .endswith() 檢查副檔名，更安全
    if file.endswith(".mp4"): 
        
        # <--- 修正1: 組合出正確的完整檔案路徑
        video_path = os.path.join(input_dir, file)
        print(f"--- 正在處理影片: {video_path} ---")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"錯誤: 無法開啟影片 {video_path}")
            continue # 跳過這個檔案，繼續下一個

        keypoints_list = []

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.2, # 您已調低，很好
            min_tracking_confidence=0.2
        ) as pose:

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    keypoints = []
                    for lm in results.pose_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                    keypoints_list.append(keypoints)

                    # 視覺化骨架
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                
                frame_count += 1
                cv2.imshow('Pose Detection', frame)
                
                # 按 ESC 可退出
                if cv2.waitKey(1) & 0xFF == 27: 
                    exit_flag = True # <--- 優化1: 設定旗標
                    break

        cap.release()

        # 儲存結果
        if len(keypoints_list) > 0:
            keypoints_array = np.array(keypoints_list)
            # <--- 修正2: 使用 os.path.splitext 取得主檔名
            base_filename = os.path.splitext(file)[0]
            output_file = os.path.join(output_dir, f"{base_filename}_pose.npy")
            np.save(output_file, keypoints_array)

            print(f"已擷取 {len(keypoints_list)} 幀骨架資料，並存成：{output_file}")
            print(f"資料 shape = {keypoints_array.shape}\n")
        else:
            print(f"在影片 {file} 中未偵測到任何骨架資料。\n")

    # <--- 優化1: 檢查旗標，如果為 True 就跳出最外層迴圈
    if exit_flag:
        print("使用者按下 ESC，程式提前結束。")
        break

cv2.destroyAllWindows()