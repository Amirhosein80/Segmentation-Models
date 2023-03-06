import torch
import torch.nn as nn
import copy
from typing import Optional, L


class ModelEma(nn.Module):
    """
    Model Exponential Moving Average
    Code from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional = None):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(self._update_fn(ema_v, model_v))

    def _update_fn(self, e, m):
        return self.decay * e + (1. - self.decay) * m

    def update(self, model):
        self._update(model)

    def set(self, model):
        self._update(model)
