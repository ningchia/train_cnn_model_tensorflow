# 使用您指定的 NGC 鏡像作為基礎
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# 避免安裝過程中的互動式提問
ENV DEBIAN_FRONTEND=noninteractive

# 1. 安裝系統層級的圖形與 OpenCV 依賴庫 (修正後的 t64 版本)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0t64 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# 2. 處理 Python 套件相容性
# 降級 NumPy 以相容 TensorFlow，並安裝特定版本的 OpenCV
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python==4.8.1.78 \
    pillow \
    matplotlib \
    tqdm \
    datasets \
    huggingface_hub

# 設定工作目錄
WORKDIR /workspace

# 預設執行指令 (可視需求更改)
CMD ["bash"]
