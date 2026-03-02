import pandas as pd
import numpy as np
import os
import glob # 用於尋找符合特定規則的檔案
import cv2  # 用於讀取影片 FPS

# --- 1. 設定所有資料夾路徑 ---
input_csv_dir = 'Training_Data'
input_pose_dir = 'pose_output'
input_video_dir = 'Training_Data' # 假設影片檔和 csv 檔在同一個資料夾
output_dir = 'combined_output'

# 建立儲存結果的資料夾
os.makedirs(output_dir, exist_ok=True)


# --- 2. 尋找所有要處理的 CSV 檔案 ---
# 使用 glob.glob 找到 'data' 資料夾下所有的 .csv 檔案路徑
csv_file_list = glob.glob(os.path.join(input_csv_dir, '*.csv'))

if not csv_file_list:
    print(f"在 '{input_csv_dir}' 資料夾中找不到任何 .csv 檔案。")
    exit()

print(f"找到 {len(csv_file_list)} 個 CSV 檔案，準備開始處理...\n")


# --- 3. 遍歷每一個 CSV 檔案進行處理 ---
for csv_file_path in csv_file_list:
    
    # --- 3.1. 從 CSV 路徑推導其他檔案的路徑 ---
    # os.path.basename 會取得檔名 (例如 'session_9.csv')
    # os.path.splitext 會將檔名和副檔名分開 (例如 ('session_9', '.csv'))
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    print(f"--- 正在處理: {base_name} ---")
    
    # 根據 base_name 組合出對應的檔案路徑
    pose_file_path = os.path.join(input_pose_dir, f"{base_name}_pose.npy")
    video_file_path = os.path.join(input_video_dir, f"{base_name}.mp4")
    output_file_path = os.path.join(output_dir, f"{base_name}_combined.csv")

    # --- 3.2. 檢查必要的檔案是否存在 ---
    if not os.path.exists(pose_file_path):
        print(f"  [警告] 找不到對應的骨架檔案: {pose_file_path}，已跳過。\n")
        continue # continue 會跳過這次迴圈，處理下一個檔案
    
    if not os.path.exists(video_file_path):
        print(f"  [警告] 找不到對應的影片檔案: {video_file_path} (無法讀取FPS)，已跳過。\n")
        continue

    # --- 3.3. 動態讀取影片的 FPS ---
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f"  [錯誤] 無法開啟影片檔案: {video_file_path}，已跳過。\n")
        cap.release()
        continue
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if video_fps == 0:
        print(f"  [警告] 影片 {video_file_path} 的 FPS 為 0，無法處理，已跳過。\n")
        continue
    
    print(f"  讀取到影片 FPS 為: {video_fps:.2f}")

    # --- 3.4. 載入資料 ---
    df_sensor = pd.read_csv(csv_file_path)
    pose_data = np.load(pose_file_path)

    # --- 3.5. 執行與之前相同的合併邏輯 ---
    df_sensor['relative_time_s'] = pd.to_numeric(df_sensor['relative_time_s'])
    df_sensor = df_sensor.sort_values('relative_time_s').reset_index(drop=True)

    df_pose = pd.DataFrame(pose_data)
    df_pose['timestamp'] = df_pose.index / video_fps

    tolerance_value = (1 / video_fps) * 0.5 

    combined_df = pd.merge_asof(
        left=df_sensor,
        right=df_pose,
        left_on='relative_time_s',
        right_on='timestamp',
        direction='nearest',
        tolerance=tolerance_value
    )

    # --- 3.6. 儲存結果與報告 ---
    combined_df.to_csv(output_file_path, index=False)
    
    last_pose_col = combined_df.columns[-1]
    nan_rows = combined_df[last_pose_col].isnull().sum()
    
    print(f"  合併完成！原始 CSV shape: {df_sensor.shape}, 骨架 shape: {pose_data.shape}")
    print(f"  合併後 shape: {combined_df.shape}")
    if nan_rows > 0:
        print(f"  注意：有 {nan_rows} 筆 CSV 資料未能成功匹配。")
    print(f"結果已儲存至: {output_file_path}\n")

print("--- 所有檔案處理完畢 ---")