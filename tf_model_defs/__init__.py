# tf_model_defs/__init__.py

# 從 tf_cnn_models 模組導入主要的模型類別
# 這樣外部腳本就可以直接從 'tf_model_defs' 包中導入這些類別
from .tf_cnn_models import create_clean_cnn_model
from .tf_cnn_models import create_mobilenet_transfer_model

# ----------------------------------------------------
# 建議: 設置 __all__ 列表 (可選，但推薦)
# 這樣可以明確指定當使用者執行 'from tf_model_defs import *' 時，應該導入哪些內容
__all__ = [
    "create_clean_cnn_model",
    "create_mobilenet_transfer_model",
]