import os
import shutil
import random
from tqdm import tqdm 

# --- 設定參數 ---
DATASET_DIR = "dataset"        # 原始資料集的根目錄
VALIDATION_SPLIT = 0.20        # 驗證集的比例，0.20 代表 20%
CLASS_NAMES = ["nothing", "hand", "cup"] 
ACTION_MODE = 'copy' # 預設使用 'copy' 複製檔案
# ------------------

# 新增：定義 Keras 標準輸出目錄
OUTPUT_DIR = "data_split_keras" # <--- 使用新的目錄名稱以區別
TRAIN_ROOT = os.path.join(OUTPUT_DIR, "train")
VALIDATE_ROOT = os.path.join(OUTPUT_DIR, "validate")
# -----------------------------------

def process_file(source_file, dest_file, mode):
    """根據模式 (copy 或 move) 處理檔案。"""
    if mode == 'copy':
        shutil.copy2(source_file, dest_file) 
    elif mode == 'move':
        shutil.move(source_file, dest_file)
    else:
        raise ValueError(f"不支援的處理模式: {mode}。必須是 'copy' 或 'move'。")


def split_dataset(base_dir, class_names, split_ratio, action_mode):
    """
    將指定目錄下的每個類別資料夾按比例分割為訓練集和驗證集，並輸出為 Keras 標準結構。
    """
    print(f"--- 資料集分割程式啟動 (Keras 模式) ---")
    
    # 建立新的根輸出目錄，並確保子目錄存在
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # 清除舊的結構
    os.makedirs(TRAIN_ROOT)
    os.makedirs(VALIDATE_ROOT)
    
    total_files_processed = 0
    
    # 遍歷每個類別
    for class_name in class_names:
        source_dir = os.path.join(base_dir, class_name)
        
        # 定義 Keras 標準目標路徑: <root>/<split_type>/<class>
        train_path = os.path.join(TRAIN_ROOT, class_name)
        validate_path = os.path.join(VALIDATE_ROOT, class_name)
        
        # 創建類別子目錄
        os.makedirs(train_path)
        os.makedirs(validate_path)
        
        print(f"\n處理類別: {class_name}")
        
        # 取得所有檔案列表 (只考慮 .jpg 檔案)
        all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        random.shuffle(all_files) 
        
        num_total = len(all_files)
        if num_total == 0:
            print(f"  --> 警告: {source_dir} 中沒有找到圖片，跳過。")
            continue
            
        # 計算分割點
        num_validate = int(num_total * split_ratio)
        num_train = num_total - num_validate
        
        print(f"  總數: {num_total} | 訓練集: {num_train} | 驗證集: {num_validate}")
        
        # --- 處理訓練集檔案 ---
        print(f"  正在處理訓練集 ({num_train} 檔案)...")
        for file_name in tqdm(all_files[num_validate:], desc=f"Train - {class_name}", leave=False):
            source_file = os.path.join(source_dir, file_name)
            dest_file = os.path.join(train_path, file_name)
            process_file(source_file, dest_file, action_mode)
            total_files_processed += 1
            
        # --- 處理驗證集檔案 ---
        print(f"  正在處理驗證集 ({num_validate} 檔案)...")
        for file_name in tqdm(all_files[:num_validate], desc=f"Validate - {class_name}", leave=False):
            source_file = os.path.join(source_dir, file_name)
            dest_file = os.path.join(validate_path, file_name)
            process_file(source_file, dest_file, action_mode)
            total_files_processed += 1

    print("\n--- 分割完成 ---")
    print(f"所有檔案已成功分割並複製到 **{OUTPUT_DIR}** 目錄下的 Keras 標準結構。")
    print(f"總共處理了 {total_files_processed} 個檔案。")

# 執行函式
try:
    split_dataset(DATASET_DIR, CLASS_NAMES, VALIDATION_SPLIT, ACTION_MODE)
except FileNotFoundError as e:
    print(f"\n錯誤: 找不到資料集目錄或其中的某些檔案。請確認路徑是否正確：{e}")
except Exception as e:
    print(f"\n發生未知錯誤: {e}")