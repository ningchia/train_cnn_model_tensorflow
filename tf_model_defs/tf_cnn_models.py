import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# --- 1. 模型定義：CleanCNN ---
# 對應 PyTorch 中的 cnn_models.CleanCNN
def create_clean_cnn_model(input_shape, num_classes=3):
    """
    創建一個模仿 PyTorch CleanCNN 結構的 Keras Sequential 模型。
    - 結構: Conv2D -> BatchNorm -> ReLU -> MaxPool2D
    - 激活函數使用 'relu'
    - 使用 GlobalAveragePooling2D 替代 AdaptiveAvgPool2d(1) + Flatten
    """
    
    # Keras 習慣使用 (Height, Width, Channels) 作為輸入形狀
    
    model = keras.Sequential([
        # 輸入層 (必須明確定義)
        keras.Input(shape=input_shape), 
        
        # 第一組 (3ch->16ch)
        layers.Conv2D(16, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(pool_size=2),
        
        # 第二組 (16->32)
        layers.Conv2D(32, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(pool_size=2),
        
        # 第三組 (32->64)
        layers.Conv2D(64, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        # PyTorch 版本移除了 MaxPool2D，並使用 AdaptiveAvgPool2d(1)
        
        layers.GlobalAveragePooling2D(), # 替代 AdaptiveAvgPool2d(1) + Flatten
        
        # 全連接分類器
        layers.Dense(num_classes, activation='softmax')
    ], name="CleanCNN")

    return model

# --- 2. 模型定義：MobileNetV2 遷移學習模型 ---
# 對應 PyTorch 中的 cnn_models.MobileNetTransfer
def create_mobilenet_transfer_model(input_shape, num_classes, use_pretrained=True):
    """
    創建一個使用 MobileNetV2 進行遷移學習的 Keras 模型。
    """
    
    if use_pretrained:
        print("MobileNetTransfer: 使用 ImageNet 預訓練權重進行初始化。")
        weights = 'imagenet'
    else:
        print("MobileNetTransfer: 權重將隨機初始化。")
        weights = None
        
    # 載入基礎模型
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, # 不包含頂部的全連接分類層
        weights=weights,   # 載入權重或None
        pooling='avg'      # 在特徵提取器末尾添加 Global Average Pooling
    )
    
    # 凍結基礎模型的所有層 (如果使用預訓練權重)
    if use_pretrained:
        base_model.trainable = False
        print("所有 MobileNetV2 基礎層已凍結。")
    
    # 建立新的分類器頭部
    x = base_model.output
    x = layers.Dropout(0.2)(x) 
    # Global Average Pooling 已由 base_model 處理 (pooling='avg')
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # 組合模型
    model = keras.Model(inputs=base_model.input, outputs=outputs, name="MobileNetTransfer")
    
    return model

# 如果你選擇使用 MobileNetV2，請在 tf_train_cnn.py 中替換模型的初始化
# 範例輸入形狀：(224, 224, 3)