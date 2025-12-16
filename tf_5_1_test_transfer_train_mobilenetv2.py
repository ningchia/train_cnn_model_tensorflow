import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from typing import List
import json

# --- 1. 配置與參數設定 (與訓練腳本保持一致) ---
MODEL_SAVE_PATH = "trained_model_tf"
CHECKPOINT_FILE = "latest_checkpoint_mobilenet.keras"   # 檢查點檔案名稱
CLASS_INDICES_FILE = "class_indices_mobilenet.json"     # <--- 類別索引檔案名稱

IMAGE_SIZE = (224, 224) # MobileNetV2 標準輸入尺寸 (H, W)
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

# 由於即時推論腳本無法直接存取訓練時的類別名稱，我們需要手動定義。
# **重要：請根據您實際訓練的類別名稱和順序進行修改！**
# 範例假設您訓練了 3 個類別：
CUSTOM_CLASSES: List[str] = ['hand', 'cup', 'nothing'] 

# 設置 GPU 記憶體增長 (與訓練腳本一致)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- 2. 輔助函式：載入模型 ---
def load_trained_model(path: str) -> tf.keras.Model:
    """載入訓練好的 Keras 模型。"""
    if not os.path.exists(path):
        print(f"❌ 找不到模型檔案: {path}")
        print("請先運行訓練腳本並確保模型已儲存。")
        exit()

    # Keras 載入模型
    model = load_model(path)
    print(f"✅ Keras 模型 '{path}' 載入完成。")
    # 將模型設置為推論模式 (確保 Dropout 關閉，BatchNorm 使用運行統計)
    # Keras/TF 模型在 predict() 時會自動進入推論模式，但顯示設置一下更明確
    # model.trainable = False # 僅為確保，predict() 會自動處理
    return model

# --- 3. 影像預處理函式 ---
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    將 OpenCV 幀 (BGR 格式) 轉換為 MobileNetV2 Keras 模型的輸入張量。
    """
    # 1. BGR -> RGB 轉換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. 縮放圖像到模型輸入尺寸
    resized_frame = cv2.resize(rgb_frame, IMAGE_SIZE)
    
    # 3. 轉換為 NumPy 陣列並添加 Batch 維度 (H, W, C -> 1, H, W, C)
    input_array = np.expand_dims(resized_frame, axis=0)
    
    # 4. MobileNetV2 預處理 (將 [0, 255] 轉換為 [-1, 1])
    # 這是與 tf_5_transfer_train_mobilenetv2.py 中 ImageDataGenerator 
    # 使用的 preprocessing_function 相同的步驟。
    processed_input = preprocess_input(input_array)
    
    return processed_input

# --- 4. 輔助函式：載入並排序類別名稱 ---
def load_class_names_from_json(model_save_path: str, filename: str = CLASS_INDICES_FILE) -> List[str]:
    """
    從 JSON 檔案中載入 class_indices 字典，並將其轉換為有序的類別名稱列表 (List)。
    回傳的列表長度即為類別數量。
    """
    json_path = os.path.join(model_save_path, filename)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ 找不到類別索引檔案: {json_path}. 請檢查訓練腳本是否已運行。")
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            class_indices = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ JSON 檔案格式錯誤: {json_path}, 錯誤: {e}")

    # 根據索引值重新排序類別名稱，以確保順序一致
    num_class_detected = len(class_indices)
    class_names = [''] * num_class_detected 
    
    for name, index in class_indices.items():
        if isinstance(index, int) and 0 <= index < num_class_detected:
            class_names[index] = name
        else:
            raise ValueError(f"類別索引 {name}: {index} 無效或超出範圍。")

    print(f"✅ 成功載入類別順序: {class_names} (共 {num_class_detected} 個)")
    return class_names

# --- 5. 主推論函式 ---
def main():
    try:
        custom_classes = load_class_names_from_json(MODEL_SAVE_PATH) # <--- 動態載入類別列表

        # 步驟 1: 載入模型
        model = load_trained_model(MODEL_PATH)
        
        # 步驟 2: 啟動 WebCam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("無法打開 WebCam。請檢查相機連接或驅動程式。")

        print("\n--- Keras 即時遷移學習模型推論已啟動 ---")
        print("按下 'q' 鍵退出。")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # --- 影像預處理 ---
            processed_input = preprocess_frame(frame)

            # --- 推論 ---
            # 使用 model.predict() 運行推論
            # Keras 會自動運行在推論模式
            outputs = model.predict(processed_input, verbose=0)
            
            # --- 結果解碼 ---
            # 輸出是機率分佈 (Softmax 結果)
            probabilities = outputs[0]
            
            # 找出最大機率的索引和置信度
            predicted_index = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
            # 獲取類別名稱
            predicted_class = custom_classes[predicted_index]
            confidence_percent = confidence * 100

            # --- 顯示結果 (使用 OpenCV) ---
            text = f"Class: {predicted_class} | Conf: {confidence_percent:.2f}%"
            
            # 將結果繪製到原始畫面 (BGR 格式)
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Real-time Custom Inference (Keras)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\n[致命錯誤] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 釋放資源
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()