import tensorflow as tf
import numpy as np
import os
import time
import json

# --- 1. 配置與參數設定 (依據現有設定) ---
MODEL_SAVE_PATH = "trained_model_tf"
FP32_MODEL_FILE = "latest_checkpoint_cifar10_mobilenet.keras" #
INT8_MODEL_FILE = "latest_checkpoint_cifar10_mobilenet_int8.tflite" # 使用 tf_tflite_converter.py 產出的檔名

IMAGE_SIZE = (224, 224)
# ImageNet 標準化參數
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- 2. 數據預處理 (與訓練時完全一致) ---
def preprocess_image(image_uint8):
    # 轉為 float32 並歸一化到 [0, 1]
    image = image_uint8.astype(np.float32) / 255.0
    # Resize 並進行 ImageNet 標準化
    image = tf.image.resize(image, IMAGE_SIZE).numpy()
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(image, axis=0) # 增加 Batch 維度

# --- 3. TFLite 推論輔助函式 ---
def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 取得模型要求的輸入資料類型
    target_dtype = input_details[0]['dtype']

    # --- 1. 處理輸入量化 ---
    # 同時支援 np.int8 與 np.uint8
    if target_dtype in [np.int8, np.uint8]:
        scale, zero_point = input_details[0]['quantization']
        # 量化公式: q = (f / scale) + zero_point
        input_data = (input_data / scale + zero_point)
        # 根據模型要求強制轉型
        input_data = input_data.astype(target_dtype)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # --- 2. 處理輸出反量化 ---
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_dtype = output_details[0]['dtype']
    
    if output_dtype in [np.int8, np.uint8]:
        scale, zero_point = output_details[0]['quantization']
        # 反量化公式: f = (q - zero_point) * scale
        output_data = (output_data.astype(np.float32) - zero_point) * scale
        
    return output_data

# --- 4. 主測試邏輯 ---
def main():
    print("--- 啟動 TensorFlow 模型量化測試 (FP32 vs. INT8 TFLite) ---")
    
    # 載入資料集
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.flatten()
    
    # 限制測試數量以節省時間 (例如取前 500 張)
    NUM_TEST = 500
    x_test, y_test = x_test[:NUM_TEST], y_test[:NUM_TEST]

    # A. 載入 FP32 模型
    fp32_path = os.path.join(MODEL_SAVE_PATH, FP32_MODEL_FILE)
    print(f"⏳ 載入 FP32 模型: {fp32_path}...")
    model_fp32 = tf.keras.models.load_model(fp32_path)

    # B. 載入 INT8 TFLite 模型
    int8_path = os.path.join(MODEL_SAVE_PATH, INT8_MODEL_FILE) # 依據儲存路徑調整
    print(f"⏳ 載入 INT8 TFLite 模型: {int8_path}...")
    interpreter = tf.lite.Interpreter(model_path=int8_path)
    interpreter.allocate_tensors()

    # 測試開始
    fp32_correct = 0
    int8_correct = 0
    fp32_total_time = 0
    int8_total_time = 0

    print(f"🚀 開始推論測試 (樣本數: {NUM_TEST})...")

    for i in range(NUM_TEST):
        # 預處理
        input_data = preprocess_image(x_test[i])
        label = y_test[i]

        # 測試 FP32
        start = time.perf_counter()
        pred_fp32 = model_fp32.predict(input_data, verbose=0)
        fp32_total_time += (time.perf_counter() - start)
        if np.argmax(pred_fp32) == label:
            fp32_correct += 1

        # 測試 INT8 TFLite
        start = time.perf_counter()
        pred_int8 = run_tflite_inference(interpreter, input_data)
        int8_total_time += (time.perf_counter() - start)
        if np.argmax(pred_int8) == label:
            int8_correct += 1

        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{NUM_TEST} 筆...")

    # --- 輸出結果 ---
    print("\n" + "=" * 45)
    print("      🔥 TensorFlow 量化效果分析報告 🔥")
    print("=" * 45)
    
    print(f"** 準確度 (Top-1 Accuracy) **")
    print(f"FP32 (.keras) 準確度: {(fp32_correct/NUM_TEST)*100:.2f}%")
    print(f"INT8 (.tflite) 準確度: {(int8_correct/NUM_TEST)*100:.2f}%")
    print(f"準確度損失: {((fp32_correct - int8_correct)/NUM_TEST)*100:.2f}%")
    
    print(f"\n** 推論速度 (平均單張耗時) **")
    print(f"FP32 平均推論時間: {(fp32_total_time/NUM_TEST):.4f} 秒")
    print(f"INT8 平均推論時間: {(int8_total_time/NUM_TEST):.4f} 秒")
    
    speed_up = fp32_total_time / int8_total_time if int8_total_time > 0 else 0
    print(f"INT8 相較於 FP32 的加速比: {speed_up:.2f} 倍")
    print("=" * 45)

if __name__ == "__main__":
    main()