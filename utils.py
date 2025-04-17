import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
def get_device():
    # 1) CUDA first
    if torch.cuda.is_available():
        return torch.device('cuda')
    # 2) then Apple MPS
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device('mps')
    # 3) fallback to CPU
    return torch.device('cpu')

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
        elif metric < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.counter = 0

class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.ema = model.clone().eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            for e_p, m_p in zip(self.ema.parameters(), model.parameters()):
                e_p.data.mul_(self.decay).add_(m_p.data, alpha=1 - self.decay)