import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim 
from PIL import Image
import pandas as pd

# ====== 1. 自定義 Dataset ======
class MultiLabelDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = self.data['filename'].values
        self.labels = self.data.iloc[:, 1:].values.astype(float)
        self.classes = self.data.columns[1:].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

# ====== 2. 設定 ======
csv_path = "/home/NAS/homes/chlunchen-10030/DeepLearning_assignment/competition2/labels.csv"
image_dir = "/home/NAS/homes/chlunchen-10030/DeepLearning_assignment/competition2/ALL"
num_epochs = 50
batch_size = 8
learning_rate = 1e-4

# ====== 3. 圖片轉換 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ====== 4. 載入資料 ======
dataset = MultiLabelDataset(csv_path, image_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = len(dataset.classes)

# ====== 5. 建立模型（ResNet18） ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ====== 6. 訓練設定 ======
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ====== 7. 訓練 loop ======
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}")

# ====== 8. 儲存模型 ======
torch.save(model.state_dict(), "multi_label_model.pth")
print("✅ 模型已儲存為 multi_label_model.pth")
