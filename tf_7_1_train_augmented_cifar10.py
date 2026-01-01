import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import numpy as np
import os
import json
import time
from datasets import load_dataset # 需要安裝: pip install datasets (Hugging Face's datasets 庫)
from PIL import Image

# --- 1. 配置與參數設定 ---
MODEL_SAVE_PATH = "trained_model_tf_augmented"
CHECKPOINT_FILE = "latest_checkpoint_augment_cifar10_by_imagenet.keras"
CLASS_INDICES_FILE = "class_indices_cifar10.json"

NUM_CLASSES = 10
NUM_EPOCHS = 50
BATCH_SIZE = 32
TRANSFER_LEARNING_LR = 0.001
IMAGE_SIZE = (224, 224)
START_MONITORING_EPOCH = 30

# ImageNet 標準化參數
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DOG_CLASS_INDEX = 5 # CIFAR-10 中狗的索引是 5

'''
# --- 2. 從 Hugging Face 抓取額外的狗圖片 (Streaming 模式) ---
# 從 ImageNet-1k (需要登入)
#  網頁授權：前往 Hugging Face ImageNet-1k 頁面，登入後點擊 "Access repository"。
#  取得 Token：在個人設定的 Access Tokens 產生一個 Read 權限的 Token。
#  程式登入：
#   from huggingface_hub import login
#   login("您的_HF_TOKEN")
def fetch_extra_dogs_from_hf(target_count=5000):
    print(f"正在從 Hugging Face 串流下載 {target_count} 張狗的圖片...")
    # ImageNet-1k 中，索引 151 到 268 大多是各種品種的狗
    DOG_LABELS_IMAGENET = list(range(151, 269)) 
    
    # 使用 streaming=True 避免下載整個 150GB 的 ImageNet
    ds = load_dataset("imagenet-1k", split="train", streaming=True, trust_remote_code=True)
    
    extra_images = []
    count = 0
    
    for example in ds:
        if example['label'] in DOG_LABELS_IMAGENET:
            img = example['image'].convert('RGB').resize((32, 32)) # 縮小到與 CIFAR-10 一致
            extra_images.append(np.array(img))
            count += 1
            if count % 500 == 0:
                print(f"已收集 {count}/{target_count} 張...")
            if count >= target_count:
                break
                
    return np.array(extra_images, dtype=np.uint8)
'''

# --- 2. 從 Hugging Face 抓取額外的狗圖片 (免登入版本：Cats vs Dogs) ---
def fetch_extra_dogs_from_hf(target_count=5000):
    print(f"正在從 Hugging Face (Cats vs Dogs) 串流讀取 {target_count} 張狗的圖片...")
    
    # 使用 microsoft/cats_vs_dogs，這個資料集不需要申請權限
    # label 0 是 cat, label 1 是 dog
    ds = load_dataset("microsoft/cats_vs_dogs", split="train", streaming=True, trust_remote_code=True)
    
    extra_images = []
    count = 0
    
    for example in ds:
        # 標籤 1 代表狗
        if example['labels'] == 1: 
            try:
                # 確保是 RGB 格式
                img = example['image'].convert('RGB')
                # 縮小到 32x32 以符合 CIFAR-10 原始規格
                img = img.resize((32, 32)) 
                extra_images.append(np.array(img))
                count += 1
                
                if count % 500 == 0:
                    print(f"已收集 {count}/{target_count} 張狗圖片...")
                
                if count >= target_count:
                    break
            except Exception as e:
                # 略過損壞的圖片
                continue
                
    return np.array(extra_images, dtype=np.uint8)

# --- 3. 數據預處理 (保持與原程式一致) ---
def preprocess_data(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# --- 4. 數據加載與「合併 & 洗牌」邏輯 ---
def get_augmented_dataset():
    # A. 載入原始 CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # 從 keras.datasets.cifar10.load_data() 下載資料時，y_train 的原始形狀是 (50000, 1)。
    # 這是一個 二維陣列（矩陣），雖然它只有一欄，但它被視為「50,000 列、1 欄」
    # 等一下用 np.full 或一般清單產生的 extra_dogs_y 通常是 一維陣列（向量），形狀為 (5000,)。
    # 所以這裡我們先把 y_train 和 y_test 攤平成一維陣列，形狀變成 (50000,) 和 (10000,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # B. 取得額外的狗圖片
    extra_dogs_x = fetch_extra_dogs_from_hf(5000)
    # 用 np.full 或一般清單產生的 extra_dogs_y 通常是 一維陣列（向量），形狀為 (5000,)。
    extra_dogs_y = np.full((len(extra_dogs_x),), DOG_CLASS_INDEX) # 標籤全部設為 5

    # C. 合併數據 (Concatenate)
    x_train_combined = np.concatenate([x_train, extra_dogs_x], axis=0)
    y_train_combined = np.concatenate([y_train, extra_dogs_y], axis=0)

    # D. 同步隨機洗牌 (Shuffle) - 使用 permutation
    print("正在對合併後的數據進行同步洗牌...")
    # permutation 生成一個從 0 到 N-1 的隨機排列數組。
    idx = np.random.permutation(len(x_train_combined))
    # 利用 NumPy 的「高級索引 (Advanced Indexing)」功能，依照剛才生成的隨機順序重新排列陣列。
    x_train_combined = x_train_combined[idx]
    y_train_combined = y_train_combined[idx]

    # E. 儲存類別字典
    class_indices = {name: i for i, name in enumerate(CIFAR10_CLASSES)}
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    with open(os.path.join(MODEL_SAVE_PATH, CLASS_INDICES_FILE), 'w') as f:
        json.dump(class_indices, f, indent=4)

    # F. 轉換為 tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train_combined, y_train_combined))
    train_ds = train_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

# --- 5. 建立遷移學習模型 (與原程式一致) ---
def create_transfer_model():
    base_model = keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# --- 6. 主程式 (加入 Class Weight) ---
def main():
    train_ds, val_ds = get_augmented_dataset()
    model = create_transfer_model()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=TRANSFER_LEARNING_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 使用 Class Weights
    # 因為 Dog 現在有 10,000 張，其他類各 5,000 張
    # 我們讓 Dog 的權重減半 (0.5)，其餘維持 1.0，平衡損失函數.
    # class_weight 會在計算 損失函數 (Loss) 時，給予樣本數較少的類別更高的「處罰權重」。
    # 所以如果模型猜錯了「狗」，讓損失小一半。
    # 也可以 狗維持是 1.0, 其他類別為 2.0, 效果類似。
    class_weights = {i: 1.0 for i in range(NUM_CLASSES)}
    class_weights[DOG_CLASS_INDEX] = 0.5 
    print(f"套用類別權重: {class_weights}")

    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=START_MONITORING_EPOCH, verbose=1)
    ]

    print("\n--- 開始增強後的遷移學習 ---")
    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weights # 傳入計算好的權重
    )
    
    print(f"訓練耗時: {(time.time() - start_time)/60:.2f} 分鐘")
    print(f"最高驗證準確度: {max(history.history['val_accuracy']):.4f}")

if __name__ == '__main__':
    main()