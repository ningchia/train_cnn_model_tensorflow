import tensorflow as tf
import argparse
import os
import numpy as np

# --- 1. 配置與訓練時一致的參數 ---
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def representative_dataset_gen():
    """
    為 INT8 量化提供真實的 CIFAR-10 數據校準。
    優化後的校準數據生成器：
    1. 使用隨機抽樣確保類別覆蓋。
    2. 嚴格執行與『推論』一致的固定預處理。
    不要進行隨機擴增, 量化校準數據應該儘可能貼近"推論"時的數據形態
    """
    (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    
    # numpy.random.choice(a, size=None, replace=True, p=None)
    #   a：抽樣對象。可以是一個一維陣列（如 [10, 20, 30]）。也可以是一個整數，表示從 0 到 a-1 的整數範圍內抽樣。
    #   size：要抽取的樣本數量or輸出的形狀。預設為 None（只抽一個）。可以是一個數字（如 100），也可以是元組（如 (3, 4)）。
    #   replace：是否允許重複抽樣。
    #   p：每個元素被選中的機率分佈。預設是均勻分佈（每個機率都一樣）。如果要自定義，陣列長度必須與 a 相同，且總和必須為 1.

    # 隨機產生 100 個索引，確保多樣性
    num_samples = 100
    # 從 0 到 len(x_train)-1 中隨機選擇 num_samples 個索引, 不重複
    indices = np.random.choice(len(x_train), num_samples, replace=False)
    
    for i in indices:
        img = x_train[i]
        
        # --- 固定預處理流程 (必須與推論腳本 tf_8_test 保持 100% 一致) ---
        # 1. 轉 float32 並縮放 [0, 1]
        img = img.astype(np.float32) / 255.0
        # 2. Resize
        img = tf.image.resize(img, IMAGE_SIZE).numpy()
        # 3. ImageNet 標準化
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # ---------------------------------------------------------
        
        # 增加 Batch 維度 (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0).astype(np.float32)
        yield [img]

def convert_model(input_path, output_path, quant_mode='none'):
    print(f"--- 模式: {quant_mode.upper()} ---")
    
    # 載入模型
    if os.path.isdir(input_path):
        converter = tf.lite.TFLiteConverter.from_saved_model(input_path)
    else:
        # 修正: Keras 3.0+ 建議使用此方法載入 .keras 檔案
        model = tf.keras.models.load_model(input_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # --- 2. 設定優化策略 ---
    if quant_mode == 'fp16':
        print("設定 Float16 量化...")
        # 啟用一組預設的優化，包括刪除未使用的節點、融合操作，以及權重後訓練量化 (Post-training weight quantization)，
        # 將浮點數權重從 32-bit 壓縮到 8-bit，但這仍使用浮點數 I/O。
        # (相當於PTQ-D的意思, 權重量化成INT8, 但每一層的輸入輸出還是FLOAT)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 設置目標類型為 float16. 
        # 告訴轉換器，如果可行，請將模型中的所有浮點數張量（主要是"權重"）轉換為 tf.float16。
        # 將權重從 32-bit 浮點數壓縮到 16-bit 浮點數 (Half-precision float)。
        # 由於它仍然是浮點數，因此不需要 representative_dataset 進行校準，轉換流程更簡單。
        # 精度損失小：相比 INT8，FLOAT16 的精度損失極小。
        # 壓縮率高：模型大小減半（從 32-bit 到 16-bit）。
        # I/O 保持 FLOAT32：通常，即使權重是 FLOAT16，模型的輸入和輸出仍然保持 FLOAT32，這對開發者來說最方便。
        converter.target_spec.supported_types = [tf.float16]
        
    elif quant_mode == 'int8':
        print("設定 INT8 全整數優化 (包含校準)...")
        # 啟用一組預設的優化，包括刪除未使用的節點、融合操作，以及權重後訓練量化 (Post-training weight quantization)，
        # 將浮點數權重從 32-bit 壓縮到 8-bit，但這仍使用浮點數 I/O。
        # (相當於PTQ-D的意思, 權重量化成INT8, 但每一層的輸入輸出還是FLOAT)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 設定代表性數據集
        converter.representative_dataset = representative_dataset_gen

        # 強制輸出為全整數 (對某些邊緣設備如 Edge TPU 是必須的)
        # 確保輸入/輸出都是 int8 類型
        # 告訴轉換器，目標裝置只支持 TFLite 內建的 INT8 整數操作。
        # 這會強制轉換器將圖中的所有浮點數運算都轉換為其對應的 INT8 整數版本。
        # (相當於PTQ-S的意思, 權重 量化成INT8, activation value 也量化成INT8, 每一層的輸入輸出也是INT8)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # 只有前面的話, 最開始 input 跟最後 output 可能還是 float32, 只是中間層的運作是INT8。
        # 這邊設定最開始 TFLite 模型的輸入節點必須接受 8-bit 整數數據。
        # 這意味著在運行時，您必須先將輸入數據手動量化成 INT8 才能傳給模型。
        converter.inference_input_type = tf.uint8 # 或 tf.int8，取決於硬體需求

        # 最終 TFLite 模型的輸出節點將輸出 8-bit 整數數據。運行時您需要將輸出結果反量化 (Dequantize) 回浮點數。
        converter.inference_output_type = tf.uint8

    else:
        print("不進行量化 (Float32)...")
        converter.optimizations = []

    # --- 3. 轉換 ---
    print("轉換中，請稍候...")
    try:
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ 成功儲存至: {output_path}")
        print(f"檔案大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"❌ 轉換失敗: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Keras 模型路徑 (.keras)')
    parser.add_argument('--mode', type=str, choices=['none', 'fp16', 'int8'], default='none')
    args = parser.parse_args()

    # 自動生成輸出檔名
    base_name = os.path.splitext(args.input)[0]
    out_file = f"{base_name}_{args.mode}.tflite"

    convert_model(args.input, out_file, args.mode)

if __name__ == "__main__":
    main()