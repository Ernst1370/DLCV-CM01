import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.data import load_data
from src.models import get_model
from src.utils import set_seed, get_device


def evaluate(config_path: str, model_path: str):
    config = yaml.safe_load(open(config_path))
    set_seed(config['training']['seed'])
    device = get_device()
    print(f"Using device: {device}")

    # Data
    _, val_loader, classes = load_data(config)

    # Model
    model = get_model(config, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Metrics
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"Overall Accuracy: {overall_acc:.2f}%")

    # Per-class
    cm = confusion_matrix(all_labels, all_preds)
    num_classes = len(classes)
    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = 100 * correct / total if total > 0 else 0
        print(f"{cls}: {acc:.2f}% ({correct}/{total})")

    # Plot Confusion Matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm_norm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)
