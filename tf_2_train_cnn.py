# 訓練時時若看到下面的訊息,"不需要"修正:
# 1. I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2. I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
#    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
#
# Msg#1
#   oneDNN (oneAPI Deep Neural Network Library) 是一個由 Intel 開發的高性能深度學習基礎庫。
#   這個訊息表示您的 TensorFlow 安裝正在使用 oneDNN 進行優化，以便在您的 CPU 上更快地執行深度學習操作（例如卷積、池化等）。
#   潛在影響：為了追求速度，oneDNN 有時會改變計算順序。由於浮點數計算的特性，不同的計算順序可能導致極小的數值差異（即浮點捨入誤差）。
#   如果您需要確保極高的數值穩定性和結果可重現性，您可以關閉此優化。
#   修正方法：在您的 Python 程式碼的最開始（在 import tensorflow as tf 之前）設置環境變數：
#       import os
#       # 關閉 oneDNN 庫的優化
#       os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
#   結果：關閉後，計算結果會更穩定，但某些操作的執行速度可能會變慢。
#
# Msg#2
#   這個訊息表示您當前使用的 TensorFlow 版本是預先編譯 (Pre-compiled) 的通用版本。
#   它已經利用了一些可用的 CPU 指令集（例如可能是 SSE 或 AVX）進行優化，以加速性能關鍵的操作。
#   核心提示：它偵測到您的 CPU 支援 AVX2, AVX_VNNI, FMA 等更先進的 CPU 指令集，但當前的 TensorFlow 版本在「其他操作」中並沒有完全利用這些指令集。
#   潛在優勢：如果重新編譯 TensorFlow 並啟用這些指令集，理論上可以進一步提高某些操作的計算速度。
#   對於大多數使用者來說，這條訊息可以被忽略。只有當您需要榨乾 CPU 的每一分性能時，才需要考慮修正。
#   修正方法：需要從原始碼 (Source Code) 重新編譯 TensorFlow。

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# 從新的模組導入模型結構
from tf_model_defs import create_clean_cnn_model, create_mobilenet_transfer_model

import os
import time
import numpy as np
import random
import warnings

# 忽略可能的警告
warnings.filterwarnings("ignore")

# --- 1. 配置與參數設定 ---
# TensorFlow 2.x 會自動管理設備 (GPU/CPU)
DATA_DIR = "data_split_keras"
MODEL_SAVE_PATH = "trained_model_tf"
CHECKPOINT_FILE = "latest_checkpoint.keras"  # Keras 推薦使用 .keras 格式
NUM_EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRANSFER_LEARNING_LR = 0.0001
IMAGE_SIZE = (224, 224) # 與 PyTorch 版本保持一致
INPUT_SHAPE = IMAGE_SIZE + (3,) # Keras 輸入形狀 (H, W, C)

WANT_REPRODUCEBILITY = False
SEED = 42

START_MONITORING_EPOCH = 150 # 決定 ModelCheckpoint 與 EarlyStopping 從第 150 個 Epoch 才開始監控

# 檢查 TensorFlow 是否能看到任何物理 GPU 裝置
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print(f"✅ TensorFlow 偵測到 {len(gpu_devices)} 個 GPU 裝置。")
    for i, gpu in enumerate(gpu_devices):
        print(f"   - GPU {i}: {gpu.name}")
else:
    print("❌ TensorFlow 未偵測到 GPU。")

# 若要確認每個operation被放置在哪個裝置上，請啟用以下設定: (debug-level 的 log 非常多!!)
#    # 將 '2' 設置為 'DEBUG' 級別，啟用所有裝置放置日誌
#    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#
#    # 讓 TensorFlow 打印出操作被放置在哪個裝置上
#    tf.debugging.set_log_device_placement(True)
#
# 若要手動指定模型或操作使用特定 GPU，請參考以下範例：
# 1. 使用 tf.device 明確分配到第一個 GPU
#    with tf.device('/GPU:0'):
#        model = create_your_model() # 模型定義
#        model.compile(...)
#
# 2. 如果使用 tf.distribute.Strategy (推薦用於多 GPU)
#    strategy = tf.distribute.MirroredStrategy()
#    with strategy.scope():
#        model = create_your_model()
#        model.compile(...)
#        
#    # model.fit(...)

# 設置可重現性
def set_seed(seed_value=42):
    """
    設定 TensorFlow/Keras 的所有隨機性種子和確定性運算。
    """
    print(f"設定所有隨機性的種子為 {seed_value}，並啟用確定性運算。")
    
    # 1. Python 標準庫和 NumPy
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    # 2. TensorFlow 核心種子
    # 這會影響所有 TensorFlow 操作，包括 CPU 和 GPU
    tf.random.set_seed(seed_value)
    
    # 3. 啟用確定性運算 (對應 PyTorch 的 cudnn.deterministic = True)
    # 這必須在 TensorFlow 初始化之前設置
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # 4. 確保 Keras 在多 GPU/分佈式訓練時也能保持確定性 (非必需，但更安全)
    # Keras 預設會依賴 tf.random.set_seed()

    # 5. 這是差異最大的部分。
    # PyTorch 的 DataLoader 採用 多進程（Multi-processing）進行數據加載，因此需要一個特殊的 worker_init_fn 
    # 來自定義每個工作進程的種子。
    # TensorFlow/Keras 的標準數據加載器 (tf.data 管道或 ImageDataGenerator) 通常使用 多線程（Multi-threading）
    # 或 TensorFlow 內建的數據處理機制，不需要像 PyTorch 那樣手動傳遞 worker_init_fn。
    # 它的隨機性來源於 NumPy 和 Python 的 random 模塊。
    # 只要在主程式碼中設置了前面提到的 random.seed() 和 np.random.seed()，通常就能確保結果的可重現性。

if WANT_REPRODUCEBILITY:
    set_seed(SEED)

# --- 2. 數據加載：Keras ImageDataGenerator & flow_from_directory ---
# 這是 TensorFlow/Keras 處理影像分類數據的標準方式，它會自動處理資料夾到類別的映射。

# PyTorch 版本的標準化參數 (ImageNet Standard)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_loaders(data_dir, batch_size, image_size):
    """
    使用 Keras ImageDataGenerator 處理數據增強、標準化和加載。
    注意：PyTorch 中的 CustomSplitDataset 的 'nothing-train-augmented'
    的複雜邏輯，在這裡需要手動處理目錄結構，才能讓 flow_from_directory 正確工作。
    假設 data_split 內部有 'train', 'validate' 子目錄，子目錄內有 'hand', 'cup', 'nothing'。
    """
    
    # --- 數據標準化與擴增 ---
    # PyTorch 的 Normalize 轉換在 Keras 中需要透過 rescale 和 normalization 結合實現
    # 實現：x = (x / 255.0 - mean) / std
    
    # 創建一個 Lambda 層來執行 PyTorch 樣式的標準化 (ImageNet mean/std)
    def normalize_tf_style(image_array):
        # 將 [0, 255] 圖像轉換為 [0, 1]
        image_array = image_array / 255.0
        # 執行 PyTorch 樣式的標準化
        normed_array = (image_array - MEAN) / STD
        return normed_array

    # 訓練集專用 Generator (包含數據擴增)
    train_datagen = ImageDataGenerator(
        preprocessing_function=normalize_tf_style, # 應用標準化
        horizontal_flip=True, # 隨機水平翻轉 (對應 RandomHorizontalFlip)
        # 其他擴增參數可在此添加
    )

    # 驗證集專用 Generator (只做標準化，不擴增)
    val_datagen = ImageDataGenerator(
        preprocessing_function=normalize_tf_style
    )
    # 若不想在tf_1_1_augment_nothing_data.py 裡用 torchvision.transforms來做影像前處理的話，
    # 可以直接用 Keras 的 ImageDataGenerator 來做前處理和增強.
    # 比如說原本在 tf_1_1_augment_nothing_data.py 裡使用了 torchvision.transforms.Compose 來做增強:
    #    aggressive_augmentations = transforms.Compose([
    #        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #        transforms.RandomResizedCrop(
    #            size=TARGET_SIZE, 
    #            scale=(0.85, 1.0),
    #            ratio=(0.75, 1.3333)
    #        ),
    #        transforms.RandomHorizontalFlip(p=0.5),
    #        transforms.RandomRotation(degrees=15),
    #        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
    #    ])
    # 若要改為使用 Keras 的APIs, 類似的設定部分可以藉由給ImageDataGenerator參數達成:
    # 這裡移除了 RandomResizedCrop 和 GaussianBlur (因為內建參數無法直接支持)
    #    train_datagen = ImageDataGenerator(
    #        rescale=1./255, # 轉換到 [0, 1] 範圍
    #        rotation_range=15, # 隨機旋轉 15 度
    #        width_shift_range=0.1, # 隨機水平平移 10%
    #        height_shift_range=0.1, # 隨機垂直平移 10%
    #        brightness_range=[0.7, 1.3], # 亮度變化 (1 +/- 0.3)
    #        horizontal_flip=True, # 隨機水平翻轉
    #        channel_shift_range=30, # 通道偏移 (模擬色相/飽和度變化)
    #        # preprocessing_function: 在此處傳入自定義的標準化和可能的高斯模糊函式
    #        preprocessing_function=None # 假設標準化在其他步驟完成
    #    )    

    # --- 數據加載 ---
    # flow_from_directory 會假設 data_dir 下有 'split_type' 資料夾，
    # 每個 'split_type' 資料夾內有 'class' 資料夾, 像是:
    # data_split/
    # ├── train/
    # │   ├── hand/
    # │   ├── cup/ 
    # │   └── nothing/
    # └── validate/
    #     ├── hand/   
    #     ├── cup/    
    #     └── nothing/
    
    # 訓練集加載器
    train_loader = train_datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', # 因為是多類別分類
        shuffle=True
    )
    
    # 驗證集加載器
    val_loader = val_datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'validate'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Keras flow_from_directory 會自動從資料夾名稱獲取類別數量
    num_classes = train_loader.num_classes
    
    if train_loader.samples == 0 or val_loader.samples == 0:
        raise ValueError(f"訓練集或驗證集為空。訓練集: {train_loader.samples}, 驗證集: {val_loader.samples}")
        
    print(f"總訓練樣本數: {train_loader.samples}")
    print(f"總驗證樣本數: {val_loader.samples}")
    print(f"偵測到類別數量: {num_classes}")
    
    # Keras loader 的輸出是 (images, one-hot_labels)，
    # PyTorch loader 的輸出是 (images, class_indices)，這裡需要適應
    return train_loader, val_loader, num_classes

# --- 3. 訓練流程主函式 ---
def train_model(train_loader, val_loader, total_epochs):
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    full_checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
    
    # --- 檢查點和續訓邏輯 ---
    start_epoch = 0
    initial_best_acc = 0.0
    is_resumed = False
    
    if os.path.exists(full_checkpoint_path):
        try:
            # 嘗試載入整個模型 (結構+權重+優化器狀態)
            model = keras.models.load_model(full_checkpoint_path)
            # Keras ModelCheckpoint 預設會儲存 epoch 資訊，但 Keras load_model 不會直接返回
            # 我們需要手動判斷續訓。這裡我們簡化處理，只載入模型狀態。
            print(f"\n[CHECKPOINT] 已載入檢查點，從上次結束的狀態恢復訓練。")
            # 由於 Keras 沒有直接記錄 epoch，我們將從頭開始計數，
            # 實際應用中，會從 ModelCheckpoint 的 log 中讀取。這裡從 0 開始，讓使用者手動設置。
            start_epoch = 0 # 簡化：不追蹤 epoch 數，只載入權重
            is_resumed = True
            
        except Exception as e:
            print(f"[警告] 檢查點載入失敗: {e}。將從頭開始訓練。")
            is_resumed = False
    
    # --- 模型初始化 ---
    if not is_resumed:
        # 使用 CleanCNN 作為示範
        model = create_clean_cnn_model(INPUT_SHAPE, num_classes=train_loader.num_classes)
        # 如果要使用 MobileNetV2，請替換成:
        # model = create_mobilenet_transfer_model(INPUT_SHAPE, train_loader.num_classes, use_pretrained=True)
    
    # --- 遷移學習/解凍邏輯 (如果使用 MobileNetV2 且要解凍) ---
    # PyTorch 程式碼中的 freeze_first_layers 邏輯在 Keras 中更精確地控制
    # 這裡假設如果模型是 MobileNetV2 且被載入續訓，我們可能會想解凍部分層。
    # 由於 CleanCNN 沒有解凍邏輯，此處省略，若使用 MobileNetV2 請自行在模型載入後修改 model.trainable 屬性。

    # --- 編譯模型 ---
    # Keras 的優化器和學習率需要在編譯時設定
    initial_lr = TRANSFER_LEARNING_LR if is_resumed else LEARNING_RATE
    
    # PyTorch 程式碼使用 Adam 優化器
    optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # 對應 nn.CrossEntropyLoss
        metrics=['accuracy']             # print info after each epoch : accuracy, loss, val_accuracy, val_loss. 
    )
    
    # model.summary() 會顯示模型結構和參數數量
    # 範例輸出:
    # Model: "CleanCNN"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #
    # =================================================================
    # input_1 (InputLayer)         [(None, 224, 224, 3)]     0       <-- 輸入層，無參數
    # _________________________________________________________________
    # conv2d (Conv2D)              (None, 224, 224, 16)      448     <-- 3*3*3*16 + 16 (偏置)
    # _________________________________________________________________
    # batch_normalization (BatchNo (None, 224, 224, 16)      64      <-- 不可訓練參數（移動平均/方差）
    # _________________________________________________________________
    # re_lu (ReLU)                 (None, 224, 224, 16)      0
    # _________________________________________________________________
    # max_pooling2d (MaxPooling2D) (None, 112, 112, 16)      0       <-- H/W 減半
    # _________________________________________________________________
    # ... (其他層次) ...
    # _________________________________________________________________
    # global_average_pooling2d (Gl (None, 64)                0       <-- 壓平為 64 維特徵向量
    # _________________________________________________________________
    # dense (Dense)                (None, 3)                 195     <-- 64*3 + 3 (偏置)
    # =================================================================
    # Total params: 8,435
    # Trainable params: 8,367
    # Non-trainable params: 68
    # _________________________________________________________________

    # 打印模型摘要
    model.summary()

    # --- Keras 回調函式 (Callbacks) ---
    # ModelCheckpoint: 用於在每個 epoch 後儲存模型
    # PyTorch 版本是「倒數 10 個 Epoch 啟動」的複雜邏輯，Keras 使用更標準的方式。
    # 這裡我們只儲存驗證準確度最高的模型
    keras_callbacks = [
        ModelCheckpoint(
            filepath=full_checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            start_from_epoch=START_MONITORING_EPOCH, # <--- 從第 N 個 Epoch 才開始儲存/監控
            verbose=1
        ),
        # 增加 EarlyStopping 來防止過度擬合 (PyTorch 程式中沒有，但實用)
        EarlyStopping(
            monitor='val_loss', 
            patience=20, # 容忍 20 個 epoch 內 loss 不下降
            mode='min',
            start_from_epoch=START_MONITORING_EPOCH, # <--- 從第 N 個 Epoch 才開始檢查停止條件
            verbose=1
        )
    ]
    
    print(f"\n--- 開始訓練 (總目標 Epoch: {total_epochs}, 從 Epoch {start_epoch + 1} 開始) ---")
    print(f"注意: 模型將儲存驗證準確度最高的狀態到 {full_checkpoint_path}。")

    # --- 模型訓練 ---
    # Keras 的 fit 方法會自動處理訓練循環、進度條和回調
    #
    # Epoch 運行期間，Keras 會顯示一個進度列，表示當前 Batch 的處理進度。
    # 進度列格式：[Batch 數] / [總 Batch 數] [時間] [Loss] [Metrics]
    # 例如: 200/200 [==============================] - 55s 275ms/step - loss: 0.8501 - accuracy: 0.6550 - val_loss: 0.6000 - val_accuracy: 0.7800
    # 這表示當前 epoch 已處理 200 個 batch，總共 200 個 batch，耗時 55 秒，每個 step 約 275 毫秒，當前 loss 和 accuracy，以及驗證集的 loss 和 accuracy。
    #
    # 當一個 Epoch 結束時，Keras 會計算並顯示該 Epoch 在整個訓練集和整個驗證集上的平均性能指標.
    # 例如: Epoch 00001: val_accuracy improved from -inf to 0.78000, saving model to trained_model_tf/latest_checkpoint.keras
    # 這表示在第 1 個 epoch 後，驗證準確度從負無限大提升到 0.78，並儲存了模型。
    #
    # model.fit() 方法會返回一個名為 tensorflow.keras.callbacks.History 的物件。
    # 這是最重要的輸出內容，因為它包含了訓練過程的所有數據記錄。
    # history.epoch：一個列表，記錄了所有已完成的 Epoch 索引 (例如 [0, 1, 2, ..., 299])。
    # history.history：一個字典，其中包含：
    #   鍵 (Keys)   ：所有在 model.compile() 中指定的 loss 名稱、metrics 名稱，以及它們對應的驗證集版本（前面加 val_）。
    #                 例如：'loss', 'accuracy', 'val_loss', 'val_accuracy'。
    #   值 (Values) ：一個列表，記錄了每個 Epoch 結束時該指標的數值。
    #
    # 例如
    #   # 打印所有記錄的指標名稱
    #   print(history.history.keys()) 
    #   # 輸出: dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    #
    #   # 獲取所有 Epoch 的訓練準確度
    #   train_acc = history.history['accuracy'] 
    #
    #   # 獲取所有 Epoch 的驗證損失
    #   val_loss = history.history['val_loss']
    #
    # 可以使用這些列表來繪製損失曲線和準確度曲線，這是判斷模型是否過度擬合或訓練是否收斂的標準做法。
    history = model.fit(
        train_loader,
        epochs=total_epochs,
        initial_epoch=start_epoch, # 從哪一個 epoch 開始 (用於續訓)
        validation_data=val_loader,
        callbacks=keras_callbacks,
        verbose=1
    )
    
    print("-" * 50)
    print("訓練流程結束。")
    
    # 輸出最終結果
    best_val_acc = max(history.history['val_accuracy'])
    print(f"整體訓練過程中的最高驗證準確度: {best_val_acc:.4f}\n")
    
# --- 4. 執行區塊 ---
if __name__ == '__main__':
    try:
        train_loader, val_loader, num_classes_detected = get_loaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)
        
        # 開始訓練
        train_model(train_loader, val_loader, NUM_EPOCHS)
        
    except ValueError as e:
        print(f"\n[資料錯誤] {e}\n請檢查 {DATA_DIR} 目錄下的檔案是否齊全且符合 Keras `split/class/image.ext` 命名格式。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[一般錯誤] 訓練過程中發生錯誤: {e}")