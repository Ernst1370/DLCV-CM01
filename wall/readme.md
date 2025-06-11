專案結構
├── inference.py                  # 推論腳本：整合YOLOv9檢測與ViT裂縫分類，並輸出結果CSV
├── wall_crack_classifier.ipynb   # ViT模型訓練與評估的Jupyter Notebook
├── requirements.txt              # 環境依賴檔案
├── README.md                     # 本文件
├── wall                          # test 資料夾
├── submission.csv                # 推論結果
├── best.pt                       # yolo 訓練權重
└── vit_wall_crack_classifier.pth # ViT 訓練權重


1. 環境安裝
conda activate 
pip install -r requirements.txt


2. 訓練（ViT）
jupyter notebook wall_crack_classifier.ipynb
訓練完產生模型權重 vit_wall_crack_classifier.pth


3. 推論
python inference.py
輸出結果將以 CSV 格式保存於 submission.csv
