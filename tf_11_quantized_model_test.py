import tensorflow as tf
import numpy as np
import cv2
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageGrab
import argparse

# --- 1. 配置與參數設定 ---
MODEL_SAVE_PATH = "trained_model_tf"
CLASS_INDICES_FILE = "class_indices_cifar10.json"
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 判定門檻 (延續您的 Margin 邏輯)
MARGIN_THRESHOLD = 0.05
OPEN_BTN_RECT = (20, 450, 120, 40)

# --- 2. TFLite 推論輔助函式 ---
class TFLiteInference:
    def __init__(self, model_path):
        print(f"📦 正在載入 TFLite 模型: {model_path}")
        # 主要的改變在於使用 tf.lite.Interpreter 取代 tf.keras.models.load_model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        # 根據模型結構，正式在記憶體（RAM）中開闢空間，用來存放輸入資料、中間層的運算結果（Activations）以及最終輸出。
        # 這是一道必須執行的指令。如果你沒有呼叫它，後面嘗試 set_tensor（把圖片丟進去）或 invoke（執行推論）時，程式會報錯。
        self.interpreter.allocate_tensors()

        # 取得輸入與輸出張量的細節. input_details 和 output_details 是 list of dicts ，裡面包含每個輸入/輸出的資訊.
        # 即使你的模型只有一個輸入和一個輸出，它們仍然以 list[0] 的形式存在。
        # ex. self.input_details[0] (入口規格)
        # {
        #    'name': 'serving_default_input_1:0',
        #    'index': 0,                                      # 該張量在 interpreter 裡的索引, 用於 set_tensor(index, data)。
        #    'shape': array([  1, 224, 224,   3], dtype=int32), # 該張量的維度, batch, height, width, channels.
        #    'dtype': <class 'numpy.float32'>,                # 數據類型（如 numpy.float32, numpy.uint8, numpy.int8）
        #    'quantization': (0.0, 0)                         # (scale, zero_point). 非量化模型為 (0.0, 0). 
        # }
        # ex. self.output_details[0] (出口規格)
        # {
        #    'name': 'StatefulPartitionedCall:0',
        #    'index': 175,
        #    'shape': array([ 1, 10], dtype=int32),           # <-- 形狀在這裡. batch, num_classes.
        #    'dtype': <class 'numpy.float32'>,
        #    'quantization': (0.0, 0)
        # }
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 檢查是否為量化模型 (INT8 模型通常輸入為 uint8)
        self.input_dtype = self.input_details[0]['dtype']
        print(f"💡 模型輸入類型: {self.input_dtype}")

    def predict(self, pil_img):
        # A. 預處理 (與訓練一致)
        img = pil_img.convert('RGB').resize(IMAGE_SIZE)
        # torch 的 transforms 接受 pillow image. 
        # 在tensorflow 裡 pillow image 需要先轉成 numpy array 後做標準化.
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        input_data = np.expand_dims(img_array, axis=0)  # 增加 batch 維度, 變成 (1, height, width, channels)

        # B. 如果模型是 INT8 量化，可能需要校準輸入數據類型
        # 需要將剛才算好的浮點數 input_data 透過公式：Q = R / S + Z 轉成整數, Q 是量化值，R 是原始值. S 是 scale，Z 是 zero_point.
        if self.input_dtype == np.uint8:
            # TFLite 的 uint8 量化通常有 scale 和 zero_point
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
        elif self.input_dtype == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)

        # C. 執行推論
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # D. 取得輸出並轉回機率 (Softmax)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # 如果輸出是量化整數，也需要轉回浮點數
        if self.output_details[0]['dtype'] in [np.uint8, np.int8]:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        return output_data

def select_model_file():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="選擇 TFLite 模型檔",
        initialdir=MODEL_SAVE_PATH,
        filetypes=[("TFLite files", "*.tflite")]
    )
    root.destroy()
    return path

def run_logic(tflite_engine, pil_img, class_names):
    predictions = tflite_engine.predict(pil_img)
    
    # Margin 判斷邏輯
    top_indices = np.argsort(predictions)[-2:][::-1]
    top1_idx, top2_idx = top_indices[0], top_indices[1]
    top1_prob, top2_prob = predictions[top1_idx], predictions[top2_idx]
    
    margin = top1_prob - top2_prob
    if margin < MARGIN_THRESHOLD:
        return "Unknown", top1_prob * 100
    else:
        return class_names[str(top1_idx)], top1_prob * 100

def main():
    # 1. 選擇 TFLite 檔案
    tflite_path = select_model_file()
    if not tflite_path: return

    # 2. 載入類別字典
    with open(os.path.join(MODEL_SAVE_PATH, CLASS_INDICES_FILE), 'r') as f:
        class_names = {str(v): k for k, v in json.load(f).items()}

    # 3. 初始化 TFLite 引擎
    engine = TFLiteInference(tflite_path)
    win_name = f'TFLite Test: {os.path.basename(tflite_path)}'
    cv2.namedWindow(win_name)

    # 4. 滑鼠事件與變數初始化
    params = {'clicked_open': False}
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = OPEN_BTN_RECT
            if bx <= x <= bx + bw and by <= y <= by + bh: param['clicked_open'] = True
    cv2.setMouseCallback(win_name, on_mouse, params)

    last_clipboard_img = None
    current_pil_img = None
    predicted_class, confidence = None, None

    while True:
        '''
        (win11裡不work)
        # 視窗關閉偵測 (雙重檢查法)
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break
        except: break
        '''
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        
        # 處理 Open 按鈕或按鍵
        if params['clicked_open'] or key == ord('o'):
            root = tk.Tk(); root.withdraw()
            file_path = filedialog.askopenfilename()
            root.destroy()
            if file_path:
                current_pil_img = Image.open(file_path)
                predicted_class, confidence = run_logic(engine, current_pil_img, class_names)
            params['clicked_open'] = False

        # 處理剪貼簿
        try:
            cb_img = ImageGrab.grabclipboard()
            if isinstance(cb_img, Image.Image):
                if last_clipboard_img is None or cb_img.size != last_clipboard_img.size:
                    current_pil_img = cb_img
                    last_clipboard_img = cb_img
                    predicted_class, confidence = run_logic(engine, current_pil_img, class_names)
        except: pass

        # 顯示處理
        if current_pil_img:
            display_img = cv2.cvtColor(np.array(current_pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
            display_img = cv2.resize(display_img, (600, 500))
        else:
            display_img = np.zeros((500, 600, 3), dtype=np.uint8)

        # UI 繪製
        bx, by, bw, bh = OPEN_BTN_RECT
        cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), -1)
        cv2.putText(display_img, "OPEN FILE", (bx + 10, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if predicted_class:
            color = (0, 255, 0) if predicted_class != "Unknown" else (0, 0, 255)
            text = f"TFLite: {predicted_class} ({confidence:.1f}%)"
            cv2.putText(display_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(win_name, display_img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()