import tensorflow as tf
import numpy as np
import cv2
import os
from typing import Literal

# --- 1. 配置與參數設定 (與 PyTorch 腳本保持一致) ---
# 檢查 GPU 是否可用 (TensorFlow 自動管理裝置，但可以檢查狀態)
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow 偵測到 GPU 裝置。")
else:
    print("TensorFlow 使用 CPU 裝置。")

MODEL_SAVE_PATH = "trained_model_tf" # Keras 模型儲存路徑
NUM_CLASSES = 3  
CLASS_NAMES = ["nothing", "hand", "cup"] # 必須與模型訓練時的索引順序一致 (0, 1, 2)
TARGET_SIZE = (224, 224) # MobileNetV2 標準輸入尺寸

# 選擇要測試的模型 (假設訓練時使用 .keras 格式儲存)
MODEL_TO_TEST: Literal['clean_cnn', 'mobilenet_v2', 'mobilenet_v2_pretrained'] = 'clean_cnn' 

CHECKPOINT_FILE = "none"
if MODEL_TO_TEST in ['clean_cnn', 'mobilenet_v2']:
    # 假設訓練腳本儲存為最新的 .keras 格式
    CHECKPOINT_FILE = "latest_checkpoint.keras" 
# 預訓練模型不需要檢查點檔案

CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

# --- 2. ImageNet 類別映射定義 (用於 mobilenet_v2_pretrained 模式) ---
# 預設所有 ImageNet 類別都會被視為 'nothing' (索引 0)。
IMAGENET_TO_CUSTOM_MAPPING = {
    # ImageNet Index : 您的目標索引 (1: hand, 2: cup)
    # 這裡的索引只是範例，需根據實際 MobileNetV2 訓練的 ImageNet 輸出類別確認
    425: 1,  
    504: 2,  
}

# --- 3. 影像預處理函式 (取代 PyTorch Transforms.Compose) ---

# ImageNet 均值和標準差 (轉換為 tf.constant，並在通道軸上廣播)
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

# 在 torch 裡我們用 torchvision.transforms 來做預處理 (resize -> to_tensor -> normalize).
# 在 TensorFlow/Keras 裡, training時我們使用 我們手動實現相同的預處理步驟.
def normalize_image_net_tf(image_array: np.ndarray):
    """
    將 OpenCV/NumPy 陣列 (BGR, [0, 255]) 轉換為標準化 TensorFlow Tensor。
    
    Args:
        image_array: BGR 格式的 NumPy 陣列 (H, W, 3)，範圍 [0, 255]。
        
    Returns:
        標準化後的 Tensor，形狀 (1, 224, 224, 3)。 -> tensorflow : (N,H,W,C), pytorch: (N,C,H,W)
    """
    # 1. 轉換為 RGB 格式 (OpenCV 讀取為 BGR)
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # 2. 轉換為浮點數 Tensor
    # TensorFlow 的影像處理 API 大多數都期望輸入是一個 tf.Tensor 物件以便更好地利用 GPU 或特定 CPU 指令集的加速。
    input_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.float32)
    
    # 3. Resize 到目標尺寸
    input_tensor = tf.image.resize(input_tensor, TARGET_SIZE)
    
    # 4. Rescale 到 [0, 1] 範圍
    input_tensor = input_tensor / 255.0
    
    # 5. Mean/Std 標準化
    normalized_tensor = (input_tensor - IMAGENET_MEAN) / IMAGENET_STD
    
    # 6. 添加 Batch 維度
    input_batch = tf.expand_dims(normalized_tensor, axis=0)
    
    return input_batch

# --- 4. 載入模型函式 ---
def load_trained_model_tf(path, num_classes):
    
    # --- 情況 3: 載入預訓練 MobileNetV2 (1000類別) ---
    if MODEL_TO_TEST == 'mobilenet_v2_pretrained':
        # 載入 MobileNetV2 結構和 ImageNet 權重 (保留 1000 類別輸出)
        # Keras Applications 預設載入 ImageNet 權重
        model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=True,  # 包含頂部分類層 (1000 類別)
            input_shape=(224, 224, 3)
        )
        print("✅ 成功載入原始預訓練 MobileNetV2 模型 (1000 類別輸出)。")
        return model

    # --- 情況 1 & 2: 載入訓練過的模型 (3類別) ---
    if not os.path.exists(path):
        # 檢查是否為 SavedModel 格式 (資料夾) 或 .keras 檔案
        if not (os.path.isdir(path) and tf.saved_model.contains_saved_model(path)) and not path.endswith('.keras'):
             raise FileNotFoundError(f"未找到檢查點檔案或 SavedModel 資料夾: {path}")

    try:
        # Keras load_model 自動載入結構、權重和編譯配置
        model = tf.keras.models.load_model(path)
        print(f"✅ 成功載入 Keras 訓練/微調後模型: {path}")
        return model
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        # 在載入失敗時，嘗試用結構重建模型（如果模型名稱已知，但我們這裡假設結構已包含）
        raise e

# --- 5. 主推論函式 ---
def main():
    try:
        # 載入模型
        model = load_trained_model_tf(CHECKPOINT_PATH, NUM_CLASSES)
        
        # 啟動 WebCam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("無法打開 WebCam。請檢查相機連接或驅動程式。")

        print("\n--- 即時推論已啟動 ---")
        print("按下 'q' 鍵退出。")

        while True:
            # 讀取一幀畫面
            ret, frame = cap.read()
            if not ret:
                break

            # --- 影像預處理 ---
            # frame 是 BGR 格式的 NumPy 陣列 (H, W, 3)
            input_batch = normalize_image_net_tf(frame)
            
            # --- 推論 ---
            # 使用 model.predict() 進行推論
            # Keras/TF 通常使用 numpy 陣列作為輸出，不需要 torch.no_grad()
            output = model.predict(input_batch, verbose=0)
            
            # --- 結果解碼 ---
            # 輸出是 (1, N) 的 NumPy 陣列
            
            # 應用 Softmax 取得機率 (NumPy 版本)
            probabilities = tf.nn.softmax(output[0]).numpy()
            
            # 找出最大機率的索引
            predicted_index_raw = np.argmax(probabilities)
            confidence = probabilities[predicted_index_raw]
            
            # *** 針對預訓練模型的特殊處理：將 1000 個輸出映射到 3 個類別 ***
            if MODEL_TO_TEST == 'mobilenet_v2_pretrained':
                imagenet_index = predicted_index_raw
                
                # 預設為 'nothing' (索引 0)
                predicted_index_custom = IMAGENET_TO_CUSTOM_MAPPING.get(imagenet_index, 0)
                predicted_index_final = predicted_index_custom
                
            else:
                # 訓練/微調後的模型 (3 類別輸出)
                predicted_index_final = predicted_index_raw
                
            predicted_class = CLASS_NAMES[predicted_index_final]
            confidence_percent = confidence * 100

            # --- 顯示結果 (使用 OpenCV) ---
            text = f"Class: {predicted_class} | Conf: {confidence_percent:.2f}%"
            
            # 在 BGR 幀上繪製文字
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Real-time Inference (TF/Keras)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except FileNotFoundError as e:
        print(f"\n[錯誤] {e}")
        print("請確認您已執行 TensorFlow/Keras 訓練腳本並成功儲存了檢查點檔案 (.keras 或 SavedModel)。")
    except Exception as e:
        print(f"\n[致命錯誤] 推論過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 釋放資源
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()