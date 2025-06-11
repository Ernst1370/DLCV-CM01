import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
index_to_class_id = [4, 8, 9, 0, 10, 6, 7, 3, 3]
num_classes = len(index_to_class_id)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("multi_label_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

df = pd.read_csv("column.csv")
results = []

for _, row in df.iterrows():
    img_id = row['ID']
    img_path = f"/home/NAS/homes/chlunchen-10030/DeepLearning_assignment/competition2/column/{img_id}.jpg"
    print(f"📂 正在處理：{img_path}")

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ 錯誤：{e}")
        results.append("")
        continue

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output)[0]
        predicted_indices = (probs > 0.1).nonzero(as_tuple=True)[0].cpu().tolist()
        predicted_classes = [str(index_to_class_id[i]) for i in predicted_indices]
        
        mapped_ids = set(map(int, predicted_classes))  # 轉 int 並轉成集合
        print(mapped_ids)
        class_A = {0, 3, 4, 6, 8}
        class_B = {5, 7}
        class_C = {1, 9, 10}

        severity = 0
        valid_ids = set()

        if mapped_ids & class_A:
            severity = 18
            valid_ids = mapped_ids & class_A
        elif mapped_ids & class_B:
            severity = 19
            valid_ids = mapped_ids & class_B
        elif mapped_ids & class_C:
            severity = 20
            valid_ids = mapped_ids & class_C
    class_str = ",".join([str(severity)] + sorted(map(str, valid_ids)))
    results.append(class_str)

df['class'] = results
df.to_csv("submission.csv", index=False)
print("✅ 結果已儲存至 submission.csv")
