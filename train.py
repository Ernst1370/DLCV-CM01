import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
from src.models import get_model
from src.data import load_data
from src.utils import set_seed, get_device, EarlyStopping, ModelEMA


def train_epoch(model, loader, criterion, optimizer, ema=None, device=get_device()):
    print(f"Using device: {device}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if ema:
            ema.update(model)
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)
    return running_loss/total, 100.*correct/total


def validate_epoch(model, loader, criterion, device=get_device()):
    print(f"Using device: {device}")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(loader, desc='Validation', leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)
    return running_loss/total, 100.*correct/total


def main(config_path: str):
    # Load config
    config = yaml.safe_load(open(config_path))
    set_seed(config['training']['seed'])
    device = get_device()

    # Data
    train_loader, val_loader, classes = load_data(config)

    # Model
    model = get_model(config, device)

    # Criterion
    weights = config['training'].get('class_weights', None)
    if weights:
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        weight_tensor = None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=config['training'].get('label_smoothing', 0.0))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    # Scheduler
    sched_cfg = config['training']['scheduler']
    if sched_cfg == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['T_max'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3)

    # EMA and Early Stopping
    ema = ModelEMA(model, decay=config.get('ema_decay', 0.99))
    stopper = EarlyStopping(patience=config['training'].get('early_stop_patience', 5),
                            delta=config['training'].get('early_stop_delta', 0.0))

    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, config['training']['epochs']+1):
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, ema, device)
        val_loss, val_acc = validate_epoch(ema.ema, val_loader, criterion, device)

        # Step scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ema.ema.state_dict(), config['training'].get('best_model_path', 'best_model.pth'))
            print(f"Saved best model (val_acc={best_acc:.2f}%)")

        stopper(val_acc)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Save final model
    torch.save(ema.ema.state_dict(), config['training'].get('final_model_path', 'final_model.pth'))
    print("Training complete.")

if __name__ == '__main__':
    import fire
    fire.Fire(main)