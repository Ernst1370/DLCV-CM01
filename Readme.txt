執行順序:
--> label.ipynb  產出多類別的訓練資料csv
--> train.py        訓練分類CNN
--> predict.py    預測
最終上傳檔案:
submission.csv


注意事項:
可以加入新的圖片讓模型訓練
但要加在/column-train/下
裡面有多個分類，已經寫好分類的對應
(在predict.py 第9行可以看到)
是按label.csv裡的順序對應
加新圖片時，也要加進去ALL資料夾

可以調predict.py裡43行的閾值


