import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping   # remove ReduceLROnPlateau. not used.
import os
import numpy as np
import math

# 導入您定義的模型結構
from tf_model_defs import create_mobilenet_transfer_model #

# --- 1. 配置與參數設定 (與 PyTorch 版本保持一致) ---
# 設置 GPU 記憶體增長，防止一次性佔滿所有 VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # 控制 TensorFlow 在 GPU 上分配記憶體的方式，確保其按需增長 (Memory Growth)，而不是預先全部分配 (Pre-allocation).
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# 訓練參數
DATA_DIR = "data_split_keras"  # 資料集目錄
MODEL_SAVE_PATH = "trained_model_tf"
CHECKPOINT_FILE = "latest_checkpoint_mobilenet.keras" # Keras 推薦的單一檔案格式
NUM_EPOCHS = 300 
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224) # MobileNetV2 標準輸入尺寸
INPUT_SHAPE = IMAGE_SIZE + (3,)

# 遷移學習時的學習率
TRANSFER_LEARNING_LR = 0.0001 # 僅訓練分類頭
FINE_TUNE_LR = 0.00001          # 微調階段學習率
FINE_TUNE_EPOCH = 100           # 開始微調的 Epoch 數

USE_PRETRAINED = True           # 是否使用預訓練權重

PATIENCE_VALUE = 40 # EarlyStopping 容忍度增加，以避免早期震盪。
START_MONITORING_EPOCH = 20 # 從第 20 個 Epoch 開始監控 EarlyStopping

# --- 2. 數據加載與預處理 ---

# MobileNetV2 標準化參數 (來自 PyTorch 範例，需要微調以符合 Keras 的 1/255 範圍)
# Keras 的 MobileNetV2 期望輸入在 [-1, 1] 範圍，因此我們使用 Keras 內建的預處理。
# Keras 的 MobileNetV2.preprocess_input 會將 [0, 255] 轉為 [-1, 1]。

# 注意：這裡的 ImageDataGenerator 讀取的是 [0, 255] 圖片。

# 訓練集專用生成器 (包含數據擴增)
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3], # 亮度變化 (1 +/- 0.3)
    horizontal_flip=True, # 隨機水平翻轉 (對應 RandomHorizontalFlip)
    channel_shift_range=30, # 通道偏移 (模擬色相/飽和度變化)
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input # 核心：將 [0, 255] 轉為 [-1, 1]
)

# 驗證集專用生成器 (不包含隨機擴增)
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input # 核心：將 [0, 255] 轉為 [-1, 1]
)

# 讀取訓練集
train_loader = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'), # 假設訓練資料夾結構為 data_split/train/class/images
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # 輸出為 One-hot 向量
    shuffle=True # 訓練集需要打亂
)

# 讀取驗證集
val_loader = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'validate'), # 假設驗證資料夾結構為 data_split/validate/class/images
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # 驗證集不需打亂
)

NUM_CLASSES = train_loader.num_classes
print(f"總訓練樣本數: {train_loader.samples}")
print(f"總驗證樣本數: {val_loader.samples}")
print(f"偵測到類別數量: {NUM_CLASSES}")


# --- 3. 模型載入與編譯 ---

# 載入 MobileNetV2 遷移學習模型
model = create_mobilenet_transfer_model(
    input_shape=INPUT_SHAPE, 
    num_classes=NUM_CLASSES, 
    use_pretrained=USE_PRETRAINED
)

# 初始階段優化器 (只訓練頂部分類器)
# Keras 會自動忽略 base_model.trainable=False 的層。
initial_optimizer = keras.optimizers.Adam(learning_rate=TRANSFER_LEARNING_LR)

# 使用稀疏交叉熵損失 (如果 class_mode='sparse') 或 CategoricalCrossentropy (如果 class_mode='categorical')
model.compile(
    optimizer=initial_optimizer,
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

# 打印模型摘要
model.summary()

# --- 4. Keras Callbacks 定義 (實現檢查點與學習率調度) ---

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

# 檢查點：保存最佳模型權重
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy', # 監控驗證準確度
    mode='max',
    save_best_only=True,    # 只保存當前最佳的模型
    verbose=1
)

# 早停 (Early Stopping)：防止過度擬合
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE_VALUE, # 增加容忍度到 40，確保模型有足夠時間度過震盪期
    restore_best_weights=True, # 訓練結束時載入最佳權重
    start_from_epoch=START_MONITORING_EPOCH, # <--- 從第 N 個 Epoch 才開始檢查停止條件 (Keras 3.0/TF 2.16+才支援)
)

# --- 5. 訓練主流程 ---

print(f"\n--- 開始訓練 (總目標 Epoch: {NUM_EPOCHS}) ---")

# --- 階段 1: 遷移學習 (只訓練分類頭) ---
print(f"\n*** 階段 1: 凍結訓練 (僅訓練頂部分類器) ***")
print(f"學習率: {TRANSFER_LEARNING_LR}")
# 模型已經在 create_mobilenet_transfer_model 中凍結了基礎層

# 計算步數
steps_per_epoch_train = math.ceil(train_loader.samples / BATCH_SIZE)      # // 是floor division. 改成向上取整數.
steps_per_epoch_val = math.ceil(val_loader.samples / BATCH_SIZE)

history = model.fit(
    train_loader,
    steps_per_epoch=steps_per_epoch_train,
    epochs=FINE_TUNE_EPOCH, # 只跑 Fine-Tuning 階段前的 Epoch 數
    validation_data=val_loader,
    validation_steps=steps_per_epoch_val,
    callbacks=[checkpoint_callback, early_stopping_callback],
    verbose=1
)


# --- 階段 2: 微調 (Fine-Tuning) ---

if NUM_EPOCHS > FINE_TUNE_EPOCH:
    print(f"\n*** 階段 2: 進入微調 (Fine-Tuning) 階段 ***")
    
    # 載入階段 1 中保存的最佳權重，確保從最佳狀態開始微調
    print("載入階段 1 中獲得的最佳模型權重。")
    model = keras.models.load_model(checkpoint_path)
    
    # Keras 函式化 API 的解凍邏輯：解凍 base_model
    # 找到 MobileNetV2 基礎模型層並解凍
    
    # 確保 MobileNetV2 的基礎模型可以訓練
    # 可以 逐層解凍 或 整體解凍
    # 找到 top layer (top model) 並從它設置 trainable=True. 這個效果會recursively propagate到底層.
    '''
    for layer in model.layers:
        if layer.name == 'mobilenetv2_1.00_224': # MobileNetV2 基礎模型通常的名稱
            layer.trainable = True
            print(f"✅ MobileNetV2 基礎層已解凍 ({layer.name})。")
            break
    '''
    # 以我們的例子(解凍全部參數)也可以直接
    model.trainable = True
            
    # PyTorch 範例中還會設定一個更低的學習率
    fine_tune_optimizer = keras.optimizers.Adam(learning_rate=FINE_TUNE_LR)

    # 重新編譯模型以使解凍/學習率生效
    model.compile(
        optimizer=fine_tune_optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print(f"學習率已更新為: {FINE_TUNE_LR}")


    # 繼續訓練
    history_fine_tune = model.fit(
        train_loader,
        # flow_from_directory() 返回的物件（通常是 DirectoryIterator 或其子類別）是一個特殊的迭代器, 已知樣本總數 (.samples 屬性).
        # 較新版本的 Keras (TensorFlow 2.x 以後) 具備足夠的智慧，能夠檢查這個迭代器是否具有 .samples 和 .batch_size 屬性。
        # 如果這些屬性存在，Keras 會在內部自動計算出 steps_per_epoch.
        steps_per_epoch=steps_per_epoch_train,      # 以目前的tensorflow 2.x 不用下這個參數, 但這邊先保留. 
        epochs=NUM_EPOCHS, # 訓練到總目標 Epoch 數
        initial_epoch=FINE_TUNE_EPOCH, # 從上次停止的地方繼續
        validation_data=val_loader,
        validation_steps=steps_per_epoch_val,
        callbacks=[checkpoint_callback, early_stopping_callback],
        verbose=1
    )

print("-" * 50)
print("訓練流程結束。")