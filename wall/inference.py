import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import csv
from torchvision import models

# === Load Models ===
vit_model = models.vit_b_16(pretrained=False)
yolo_model = YOLO("best.pt")
vit_model.load_state_dict(torch.load("vit_wall_crack_classifier.pth"))
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, 10)
vit_model.eval()

# === Define Class Mapping ===
crack_to_criteria = {
    0: [0],        # exposed rebar
    1: [2],        # huge spalling
    2: [3],        # x/v shape
    3: [4],        # continuous diagonal
    4: [5],        # discontinuous diagonal
    5: [6],        # continuous vertical
    6: [7],        # discontinuous vertical
    7: [8],        # continuous horizontal
    8: [9],        # discontinuous horizontal
    9: [10],       # small cracks
}

criteria_to_damage = {
    0: 18, 2: 18,      # A
    3: 19, 4: 19, 6: 19, 8: 19,    # B
    5: 20, 7: 20, 9: 20, 10: 20    # C
}

# === Preprocessing for ViT ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Inference Loop ===
def run_inference(image_folder, output_csv="submission.csv"):
    results_dict = {}

    for filename in sorted(os.listdir(image_folder)):
        if not filename.endswith(".jpg") and not filename.endswith(".png"):
            continue

        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        image_id = os.path.splitext(filename)[0]

        yolo_result = yolo_model(image)[0]
        labels = []
        damage_classes = set()

        for box in yolo_result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            cls = int(cls)

            if cls == 0:
                labels.append(0)
                damage_classes.add(criteria_to_damage[0])
            elif cls == 1:
                labels.append(2)
                damage_classes.add(criteria_to_damage[2])
            elif cls == 2:
                # Crack: crop and classify
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(pil_crop).unsqueeze(0)
                with torch.no_grad():
                    crack_pred = vit_model(input_tensor).argmax().item()
                crit_labels = crack_to_criteria[crack_pred]
                labels.extend(crit_labels)
                for c in crit_labels:
                    if c in criteria_to_damage:
                        damage_classes.add(criteria_to_damage[c])

        # Choose the worst damage level (max of 18,19,20)
        if damage_classes:
            damage_label = max(damage_classes)
        else:
            damage_label = 18  # Default if no damage found

        full_label = [damage_label] + sorted(set(labels))
        results_dict[image_id] = full_label

    # Write CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["ID", "class"])
        for img_id in sorted(results_dict.keys(), key=lambda x: int(x)):
            label_str = "{}".format(",".join(str(x) for x in results_dict[img_id]))
            writer.writerow([img_id, label_str])

    print(f"CSV saved to {output_csv}")

image = cv2.imread("C:/Users/USER/Downloads/datasets/test_data/wall/4.jpg")
results = yolo_model(image)[0]
test_folder = "wall"
os.makedirs(test_folder, exist_ok=True)
run_inference(test_folder, output_csv="submission.csv")