專案結構
├── inference.py                   # 推論腳本：整合 YOLOv9 檢測與 ViT 裂縫分類
├── wall_crack_classifier.ipynb    # ViT 模型訓練與評估的 Notebook
├── wall_resize/                   # 訓練資料集資料夾（請從 Google Drive 下載）
├── wall/                          # 測試資料集資料夾（請從 Google Drive 下載）
├── vit_wall_crack_classifier.pth  # ViT 訓練權重（請從 Google Drive 下載）
├── best.pt                        # YOLOv9 訓練權重（請從 Google Drive 下載）
├── submission.csv                 # 推論輸出 CSV
├── requirements.txt               # 套件依賴
└── README.md                      # 本文件


下載所有Google Drive檔案
https://drive.google.com/drive/folders/1npKj6zgDbDhSJ09bYqEShJHRp-Baan9u?usp=drive_link


環境安裝
conda create -n wall python=3.10
conda activate wall
pip install -r requirements.txt


訓練（ViT）
jupyter notebook wall_crack_classifier.ipynb
訓練完產生模型權重 vit_wall_crack_classifier.pth


推論
python inference.py
輸出結果將以 CSV 格式保存於 submission.csv
