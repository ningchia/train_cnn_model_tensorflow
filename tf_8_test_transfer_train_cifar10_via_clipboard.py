import tensorflow as tf
import numpy as np
import cv2
import os
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageGrab
import time

# --- 1. 配置與參數設定 ---
MODEL_SAVE_PATH = "trained_model_tf"
CHECKPOINT_FILE = "latest_checkpoint_cifar10_mobilenet.keras"
CLASS_INDICES_FILE = "class_indices_cifar10.json"
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# 按鈕區域定義 (x, y, w, h)
OPEN_BTN_RECT = (20, 450, 120, 40)

# --- 2. 輔助功能：檔案瀏覽器 ---
def select_image_file():
    root = tk.Tk()
    root.withdraw() # 隱藏主視窗
    file_path = filedialog.askopenfilename(
        title="選擇測試圖檔",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()
    return file_path

# --- 3. 預處理與推論 ---
def preprocess_image(pil_img):
    # 轉為 RGB 並 Resize
    img = pil_img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    # 轉為 numpy 並歸一化到 [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    # ImageNet 標準化
    img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
    # 增加 Batch 維度 (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)

def run_inference(model, pil_img, class_names):
    input_data = preprocess_image(pil_img)
    predictions = model.predict(input_data, verbose=0)  # 輸出是 (1, N) 的 NumPy 陣列
    # 因為訓練時期的model定義最後一層是 layers.Dense(NUM_CLASSES, activation='softmax'), 代表輸出已經是softmax後的機率, 而非logits.
    # 所以這邊的predictions 已經是機率.
    
    # 取得機率最高的索引
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx]
    return class_names[str(idx)], confidence * 100

# --- 4. 繪製 UI ---
def draw_ui(frame, predicted_class=None, confidence=None):
    # 繪製按鈕
    x, y, w, h = OPEN_BTN_RECT
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), -1)
    cv2.putText(frame, "OPEN FILE", (x + 10, y + 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 繪製結果
    if predicted_class:
        text = f"Class: {predicted_class} ({confidence:.1f}%)"
        cv2.putText(frame, text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.putText(frame, "Press 'Q' to Quit | Clipboard Monitored", (20, 420), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# --- 5. 主程式 ---
def main():
    # A. 載入類別字典
    try:
        with open(os.path.join(MODEL_SAVE_PATH, CLASS_INDICES_FILE), 'r') as f:
            indices = json.load(f)
            # 反轉字典: { "0": "plane", "1": "car" ... }
            class_names = {str(v): k for k, v in indices.items()}
    except Exception as e:
        print(f"無法讀取類別索引檔: {e}")
        return

    # B. 載入模型 (TensorFlow 不需要重建結構)
    print("正在載入 Keras 模型...")
    model = tf.keras.models.load_model(CHECKPOINT_PATH)
    print("✅ 模型載入成功！")

    cv2.namedWindow('CIFAR-10 TF Inference')
    
    # 用於滑鼠事件處理
    params = {'clicked_open': False}
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = OPEN_BTN_RECT
            if bx <= x <= bx + bw and by <= y <= by + bh:
                param['clicked_open'] = True
    
    cv2.setMouseCallback('CIFAR-10 TF Inference', on_mouse, params)

    last_clipboard_img = None
    current_pil_img = None
    predicted_class, confidence = None, None

    print("🚀 推論引擎啟動。點擊視窗按鈕、按 'O' 鍵或複製圖片到剪貼簿...")

    while True:
        # 1. 檢查按鈕點擊或按鍵
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        
        if params['clicked_open'] or key == ord('o'):
            file_path = select_image_file()
            if file_path:
                current_pil_img = Image.open(file_path)
                predicted_class, confidence = run_inference(model, current_pil_img, class_names)
            params['clicked_open'] = False

        # 2. 檢查剪貼簿
        try:
            cb_img = ImageGrab.grabclipboard()
            if isinstance(cb_img, Image.Image):
                # 簡單的比對方式：判斷是否為新圖
                if last_clipboard_img is None or cb_img.size != last_clipboard_img.size:
                    print("📋 偵測到剪貼簿新圖片！")
                    current_pil_img = cb_img
                    last_clipboard_img = cb_img
                    predicted_class, confidence = run_inference(model, current_pil_img, class_names)
        except:
            pass

        # 3. 顯示畫面
        if current_pil_img:
            # 將 PIL 轉回 OpenCV 格式進行顯示
            display_img = np.array(current_pil_img.convert('RGB'))
            display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
            # 固定顯示大小
            display_img = cv2.resize(display_img, (600, 500))
        else:
            display_img = np.zeros((500, 600, 3), dtype=np.uint8)

        draw_ui(display_img, predicted_class, confidence)
        cv2.imshow('CIFAR-10 TF Inference', display_img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()