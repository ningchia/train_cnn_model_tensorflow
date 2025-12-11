import tensorflow as tf
import argparse
import os
import numpy as np

# --- 1. INT8 量化所需的代表性數據集生成器 ---
# ⚠️ 警告：
# 這是 INT8 量化 (全整數) 必需的步驟。它用於校準 (Calibration)，以確定浮點數張量到 INT8 的最佳縮放因子。
# 函式必須產出代表真實數據分佈的輸入張量。
# 在您的實際應用中，請將此處的隨機數據替換為您的訓練或驗證數據集。
# 輸入形狀必須與您模型的輸入形狀完全匹配（例如：(1, 224, 224, 3)）。
def representative_dataset_gen():
    """
    提供用於 INT8 量化校準的數據集生成器。
    這裡使用虛擬數據作為範例。
    """
    # 假設您的模型輸入形狀是 (224, 224, 3)
    INPUT_SHAPE = (1, 224, 224, 3) 
    # 只需要提供少量批次 (~100 批次)
    for _ in range(100): 
        # 產出一個批次 (這裡使用 Batch Size = 1)
        # 數據必須是浮點數 (tf.float32) 且已標準化 (例如 [0, 1] 或 Mean/Std)
        yield [np.random.rand(*INPUT_SHAPE).astype(np.float32)]

# --- 2. 轉換核心函式 ---
def convert_to_tflite(input_path: str, output_path: str, quantization_type: str):
    print(f"--- 載入模型: {input_path} ---")
    
    # 1. 創建轉換器實例
    if os.path.isdir(input_path):
        # SavedModel 資料夾
        print("檢測到 SavedModel 格式...")
        converter = tf.lite.TFLiteConverter.from_saved_model(input_path)
    elif input_path.lower().endswith(('.keras', '.h5')):
        # Keras 檔案格式 (.keras 或 .h5)
        print(f"檢測到 Keras 檔案格式 ({input_path[-5:]})...")
        try:
            # 載入模型結構與權重
            model = tf.keras.models.load_model(input_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
        except Exception as e:
            print(f"❌ 載入 Keras 模型失敗: {e}")
            return
    else:
        print("❌ 錯誤: 輸入路徑必須是有效的 .keras/.h5 檔案或 SavedModel 資料夾。")
        return

    # --- 2. 設置優化與量化配置 ---
    # 啟用一組預設的優化，包括刪除未使用的節點、融合操作，以及權重後訓練量化 (Post-training weight quantization)，
    # 將浮點數權重從 32-bit 壓縮到 8-bit，但這仍使用浮點數 I/O。
    # (相當於PTQ-D的意思, 權重量化成INT8, 但每一層的輸入輸出還是FLOAT)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if quantization_type == 'int8':
        print("⚙️ 設定: INT8 (全整數量化，需校準)。")
        
        # 設置代表性數據集 (校準)
        converter.representative_dataset = representative_dataset_gen
        
        # 確保輸入/輸出都是 int8 類型
        # 告訴轉換器，目標裝置只支持 TFLite 內建的 INT8 整數操作。
        # 這會強制轉換器將圖中的所有浮點數運算都轉換為其對應的 INT8 整數版本。
        # (相當於PTQ-S的意思, 權重 量化成INT8, activation value 也量化成INT8, 每一層的輸入輸出也是INT8)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        
        # 只有前面的話, 最開始 input 跟最後 output 可能還是 float32, 只是中間層的運作是INT8。
        # 這邊設定最開始 TFLite 模型的輸入節點必須接受 8-bit 整數數據。
        # 這意味著在運行時，您必須先將輸入數據手動量化成 INT8 才能傳給模型。
        converter.inference_input_type = tf.int8

        # 最終 TFLite 模型的輸出節點將輸出 8-bit 整數數據。運行時您需要將輸出結果反量化 (Dequantize) 回浮點數。
        converter.inference_output_type = tf.int8
        
    elif quantization_type == 'float16':
        print("⚙️ 設定: FLOAT16 (半精度浮點數量化)。")
        # 設置目標類型為 float16. 
        # 告訴轉換器，如果可行，請將模型中的所有浮點數張量（主要是"權重"）轉換為 tf.float16。
        # 將權重從 32-bit 浮點數壓縮到 16-bit 浮點數 (Half-precision float)。
        # 由於它仍然是浮點數，因此不需要 representative_dataset 進行校準，轉換流程更簡單。
        # 精度損失小：相比 INT8，FLOAT16 的精度損失極小。
        # 壓縮率高：模型大小減半（從 32-bit 到 16-bit）。
        # I/O 保持 FLOAT32：通常，即使權重是 FLOAT16，模型的輸入和輸出仍然保持 FLOAT32，這對開發者來說最方便。
        converter.target_spec.supported_types = [tf.float16]
        
    elif quantization_type == 'none':
        print("⚙️ 設定: FLOAT32 (無量化，使用預設浮點數)。")
        # 移除優化選項，確保是純 FLOAT32
        # 確保轉換器不會進行任何默認的權重壓縮或優化，強制輸出純粹的 FLOAT32 模型。
        converter.optimizations = [] 
    
    # --- 3. 轉換與儲存 ---
    print("開始模型轉換...")
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"❌ 模型轉換失敗: {e}")
        return

    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ 轉換成功! 檔案儲存至: {output_path}")
    print(f"檔案大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="將 Keras 模型 (.keras/.h5) 或 SavedModel 轉換為 TFLite 格式，可選量化優化。")
    parser.add_argument(
        '--input_path', 
        type=str, 
        required=True, 
        help='輸入模型路徑 (.keras/.h5 檔案或 SavedModel 資料夾路徑)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='converted_model.tflite', 
        help='輸出 .tflite 檔案路徑 (預設: converted_model.tflite)'
    )
    parser.add_argument(
        '--quantization_type', 
        type=str, 
        choices=['none', 'float16', 'int8'], 
        default='none', 
        help='選擇量化優化類型: none (FLOAT32), float16 (半精度), int8 (全整數)。'
    )
    
    args = parser.parse_args()
    
    convert_to_tflite(args.input_path, args.output_path, args.quantization_type)

if __name__ == '__main__':
    # 確保 TensorFlow 被導入
    try:
        import tensorflow as tf
    except ImportError:
        print("錯誤: 缺少 TensorFlow 模組。請確認環境中已安裝: pip install tensorflow")
        exit(1)
    main()