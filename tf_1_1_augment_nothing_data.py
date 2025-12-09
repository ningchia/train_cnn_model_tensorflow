import os
from torchvision import transforms
from PIL import Image
import warnings
from tqdm import tqdm
import shutil

# --- 1. 配置與參數設定 (針對 Keras 標準結構調整) ---
# Keras 分割後的根目錄
BASE_DATA_DIR = "data_split_keras" 
# 來源資料夾 (由 1_data_split.py 產生)
SOURCE_CLASS_FOLDER = "train/nothing" 
# 目標資料夾：直接是 train/nothing
TARGET_CLASS_FOLDER = "train/nothing" 
# ----------------------------------------------------

NUM_AUGMENTATIONS_PER_IMAGE = 4  # 每張圖片額外生成 4 個版本，總計擴增 5 倍
TARGET_SIZE = (224, 224) 

# 忽略 PIL/Image 庫可能發出的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 2. 定義積極的數據增廣 (Data Augmentation) 策略 (不變) ---
# ... (transforms.Compose 保持不變) ...
# 雖然是在tensorflow環境中，但我們使用 PyTorch 的 transforms 來進行增廣，因為它們更靈活且易於使用。
aggressive_augmentations = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(
        size=TARGET_SIZE, 
        scale=(0.85, 1.0),
        ratio=(0.75, 1.3333)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
])

# --- 3. 主增廣流程 ---
def augment_data():
    source_dir = os.path.join(BASE_DATA_DIR, SOURCE_CLASS_FOLDER)
    target_dir = os.path.join(BASE_DATA_DIR, TARGET_CLASS_FOLDER)

    print(f"來源資料夾 (將被覆蓋): {source_dir}")
    print(f"目標資料夾 (即來源資料夾): {target_dir}")
    
    if not os.path.isdir(source_dir):
        print(f"錯誤：找不到來源資料夾 {source_dir}。請先執行 1_data_split.py。")
        return
        
    # 暫時將原始圖片移出，確保只對原始圖片進行擴增，防止重複擴增和污染
    temp_dir = os.path.join(BASE_DATA_DIR, "temp_nothing_original")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    original_count = len(image_files)
    
    if original_count == 0:
        print(f"來源資料夾 {source_dir} 中沒有找到任何圖片。")
        return
        
    # --- 關鍵步驟：將原始圖片移到暫存區，然後清空原始目錄 ---
    print(f"正在將 {original_count} 張原始圖片移至暫存區...")
    for filename in image_files:
         shutil.move(os.path.join(source_dir, filename), os.path.join(temp_dir, filename))
    
    # 清空原始目錄，準備存儲擴增後的全部文件
    shutil.rmtree(source_dir)
    os.makedirs(source_dir)
    
    # --- 開始擴增 ---
    image_files = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_new_count = 0
    
    pbar = tqdm(image_files, desc="正在增廣圖片")
    
    for filename in pbar:
        img_path = os.path.join(temp_dir, filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告：無法讀取或轉換檔案 {filename} ({e})，跳過。")
            continue

        base_name, ext = os.path.splitext(filename)
        
        # 1. 儲存原始圖片 (作為第一個樣本)
        # 存回 TARGET_CLASS_FOLDER (即 source_dir)
        original_output_path = os.path.join(target_dir, filename)
        img.save(original_output_path)
        total_new_count += 1
        
        # 2. 生成並儲存增廣圖片
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            augmented_img = aggressive_augmentations(img)
            new_filename = f"{base_name}_aug_{i}{ext}"
            new_output_path = os.path.join(target_dir, new_filename)
            augmented_img.save(new_output_path)
            total_new_count += 1
            
    # 清理暫存資料夾
    shutil.rmtree(temp_dir)
    
    print("-" * 50)
    print(f"✅ 增廣完成！")
    print(f"原始圖片數量: {original_count} 張")
    print(f"擴增後總圖片數量: {total_new_count} 張 (原始數量的 {total_new_count / original_count:.1f} 倍)")
    print(f"所有檔案已儲存至: {target_dir}")

if __name__ == '__main__':
    augment_data()