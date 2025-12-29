import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy as np
import os
import json
import time

# --- 1. 配置與參數設定 ---
# 與 PyTorch 版本保持一致
MODEL_SAVE_PATH = "trained_model_tf"
CHECKPOINT_FILE = "latest_checkpoint_cifar10_mobilenet.keras"
CLASS_INDICES_FILE = "class_indices_cifar10.json"

NUM_CLASSES = 10
NUM_EPOCHS = 50 # 僅訓練 50 個 Epochs
BATCH_SIZE = 32
TRANSFER_LEARNING_LR = 0.001
IMAGE_SIZE = (224, 224) 
START_MONITORING_EPOCH = 30  # 從第 30 個 Epoch 開始監控 EarlyStopping

# ImageNet 標準化參數
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR-10 類別名稱
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- 2. 輔助函式：數據預處理 ---
def preprocess_data(image, label):
    # 1.轉換為 float32 並歸一化到 [0, 1]
    #   Resize 到 MobileNetV2 期望的 224x224
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)

    # 2. 幾何擴增（翻轉不影響數值範圍，隨時可以做）
    image = tf.image.random_flip_left_right(image)
    
    # 3. 數值擴增（重點：參數必須很小，因為目前是 0~1）
    # 這裡 max_delta=0.1 代表亮度隨機增減最大 10%
    image = tf.image.random_brightness(image, max_delta=0.1) 
    # 隨機對比度，參數是倍率，通常不影響量級，但仍建議放在標準化前
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # 4. 重要步驟：截斷（Clipping）
    # 確保亮度調整後，數值依然嚴格落在 [0, 1] 之間
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 5. 最後才進行 ImageNet 標準化 (減均值、除方差)
    # 經過這步後，數值會變成有正有負（例如 -2.1 到 2.3），這才是模型最喜歡的輸入
    # 執行 ImageNet 標準化
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    
    # 標籤轉為 One-hot (因模型使用 categorical_crossentropy)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# --- 3. 數據加載與類別字典儲存 ---
def get_dataset():
    print("正在載入 CIFAR-10 數據集...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    print(f"訓練影像形狀: {x_train.shape}") # 輸出: (50000, 32, 32, 3) -> (張數, 高, 寬, 通道)
    print(f"訓練標籤形狀: {y_train.shape}") # 輸出: (50000, 1)
    print(f"影像像素類型: {x_train.dtype}") # 輸出: uint8 (0-255 的整數)
    print(f"第一筆標籤內容: {y_train[0]}") # 輸出: [6]
    
    # 扁平化標籤 (cifar10 載入時是 [[label], [label]])
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # 建立與儲存類別字典，供推論腳本讀取
    class_indices = {name: i for i, name in enumerate(CIFAR10_CLASSES)}
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with open(os.path.join(MODEL_SAVE_PATH, CLASS_INDICES_FILE), 'w') as f:
        json.dump(class_indices, f, indent=4)
    print(f"✅ 類別索引已儲存至: {os.path.join(MODEL_SAVE_PATH, CLASS_INDICES_FILE)}")

    # 建立 tf.data 管道 (高效並行處理)
    # .from_tensor_slices 從記憶體或硬碟讀取數據
    # .map() 讓 CPU 多核心並行執行 tf.image 的動作（如隨機翻轉、標準化）. tf.data.AUTOTUNE 會自動分配適當的thread數來做 tf.image 處理.
    # .prefetch() 讓 CPU 在 GPU 訓練當前批次（Batch）時，就預先準備好下一個批次.
    # .shuffle(5000) 先從數據庫中取出前 5000 張圖放進一個「緩衝區（Buffer）」。
    #                從這 5,000 張圖中隨機抽出一個送去訓練。再從原始數據庫取下一張新圖補進來。重複這個動作。
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(5000).map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# --- 4. 建立遷移學習模型 ---
def create_transfer_model():
    # 1. 載入預訓練 MobileNetV2，不含頂層分類器
    base_model = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # 2. 凍結基礎層 (只訓練分類頭)
    base_model.trainable = False
    print("💡 模型基礎特徵提取層已凍結。")

    # 3. 建立自定義分類頭
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2), # 添加輕微 Dropout 防止過擬合
        layers.Dense(NUM_CLASSES, activation='softmax')     # 讓output直接是機率分佈而非Logits.讓後續model.predict不用做softmax.
    ])
    
    return model

# --- 5. 訓練主程式 ---
def main():
    train_ds, val_ds = get_dataset()
    model = create_transfer_model()

    # 編譯模型
    model.compile(
        optimizer=optimizers.Adam(learning_rate=TRANSFER_LEARNING_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 設定回調函式 (儲存最佳模型與 EarlyStopping)
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10, # CIFAR-10 收斂較快，設定容忍度 10
            restore_best_weights=True,
            start_from_epoch=START_MONITORING_EPOCH, # <--- 從第 N 個 Epoch 才開始檢查停止條件 (Keras 3.0/TF 2.16+才支援)
            verbose=1
        )
    ]

    print(f"\n--- 開始遷移學習 (總目標 Epoch: {NUM_EPOCHS}) ---")
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )
    end_time = time.time()

    print("-" * 50)
    print(f"訓練耗時: {(end_time - start_time)/60:.2f} 分鐘")
    print(f"最高驗證準確度: {max(history.history['val_accuracy']):.4f}")

if __name__ == '__main__':
    main()