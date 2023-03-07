import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.cuda.amp as amp
import datetime
import os

from utils.model_ema import ModelEma
from loss_funcs import BootstrappedCE, FocalLoss
from collections import OrderedDict


class Trainer(object):
    def __init__(self, model, config, name: str, device: str):
        self.name = name
        self.config = config
        self.device = device
        self.now = datetime.datetime.now()
        self.now = f"{self.now.year}-{self.now.month}-{self.now.day}_" \
                   f"{self.now.hour}-{self.now.minute}-{self.now.second}"
        self.folder = './checkpoint'
        os.makedirs(self.folder, exist_ok=True)

        self.model = model
        self.model.to(self.device)
        self._get_optimizer()
        self._get_criterion()
        self._resume()

    def _train(self):
        pass

    def _test(self):
        pass

    def _save(self, acc, epoch=-1):
        if acc > self.best_acc:
            print('Saving checkpoint...')
            state = {
                'model': self.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'scaler': self.scaler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }
            if self.model_ema is not None:
                state.update({
                    "ema": self.model_ema.state_dict(),
                    "model_ema": self.model_ema.module.state_dict()
                })
            path = os.path.join(os.path.abspath(self.folder), f"{self.name}_{epoch}" + '.pth')
            torch.save(state, path)
            torch.save(state, "./best.pth")
            self.best_acc = acc

    def _resume(self):
        self.start_epoch, self.best_acc = -1, 0.0
        if os.path.isfile(f"{self.name}" + '.pth'):
            try:
                loaded = torch.load(f"{self.name}" + '.pth')
                self.model.load_state_dict(loaded["model"], strict=True)
                self.optimizer.load_state_dict(loaded["optimizer"])
                self.scaler.load_state_dict(loaded["scaler"])
                self.scheduler.load_state_dict(loaded["scheduler"])
                self.start_epoch = loaded["epoch"]
                loaded = torch.load("best" + '.pth')
                self.best_acc = loaded["acc"]
                if self.model_ema is not None:
                    self.model_ema.load_state_dict(loaded["ema"])
                    self.model_ema.module.load_state_dict(loaded["model_ema"])
                print(f"loaded all parameters from best checkpoint, acc: {self.best_acc} :)")
                print()
            except:
                print("Something is wrong Train from Scratch :( ")

        else:
            print("I can't find best.pth so we can't load params :( ")

    def _get_optimizer(self):
        if hasattr(self.model, "get_params"):
            params = self.model.get_params(lr=self.config.ADAMW_LR, weight_decay=self.config.WEIGHT_DECAY)
        else:
            params = self.model.parameters()

        if self.config.OPTIMIZER == "ADAMW":
            self.optimizer = optim.AdamW(params, lr=self.config.ADAMW_LR, weight_decay=self.config.WEIGHT_DECAY,
                                         betas=self.config.ADAMW_BETAS)

        elif self.config.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(params, lr=self.config.ADAMW_LR, weight_decay=self.config.WEIGHT_DECAY,
                                       momentum=self.config.SGD_MOMENTUM)

        else:
            raise NotImplemented

        self.scaler = amp.GradScaler()

        total_iters = (self.config.NUM_TRAIN_DATAS // self.config.BATCH_SIZE) * \
                      (self.config.EPOCHS - self.config.WARMUP_EPOCHS)
        warm_iters = (self.config.NUM_TRAIN_DATAS // self.config.BATCH_SIZE) * self.config.WARMUP_EPOCHS
        num_epochs = self.config.EPOCHS - self.config.WARMUP_EPOCHS

        if self.config.SCHEDULER_METHOD == "POLY":
            main_lr_scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=total_iters,
                                                                power=self.config.POLY_POWER)

        elif self.config.SCHEDULER_METHOD == "COS":
            main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        else:
            raise NotImplemented

        if self.config.WARMUP_EPOCHS > 0:
            warm_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=self.config.WARMUP_FACTOR, total_iters=warm_iters)

            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warm_lr_scheduler, main_lr_scheduler], milestones=[warm_iters])

        else:
            self.scheduler = main_lr_scheduler

        if self.config.USE_EMA:
            self.model_ema = ModelEma(self.model, decay=self.config.EMA_DECAY, device=self.device)
        else:
            self.model_ema = None

    def _get_criterion(self):
        if self.config.LOSS == "CROSS":
            self.criterion = nn.CrossEntropyLoss(
                weight=self.config.CLASS_WEIGHTS if self.config.USE_CLASS_WEIGHTS else None,
                ignore_index=self.config.IGNORE_LABEL,
                label_smoothing=self.config.LABEL_SMOOTHING
            )
        elif self.config.LOSS == "OHEM":
            self.criterion = BootstrappedCE(loss_th=self.config.OHEM_THRESH, ignore_index=self.config.IGNORE_LABEL,
                                            label_smoothing=self.config.LABEL_SMOOTHING,
                                            weight=self.config.CLASS_WEIGHTS if self.config.USE_CLASS_WEIGHTS else None
                                            )
        elif self.config.LOSS == "FOCAL" or self.config.LOSS == "FOCAL_IOU":
            self.criterion = FocalLoss(alpha=self.config.FOCAL_ALPHA, gamma=self.config.FOCAL_GAMMA,
                                       ignore_index=self.config.IGNORE_LABEL,
                                       use_iou_loss=True if self.config.LOSS == "FOCAL_IOU" else False)
        else:
            raise NotImplemented

    def _criterion_forward(self, outputs: OrderedDict, target: torch.Tensor):
        semantic_loss = 0
        semantic_aux = 0
        for k, v in outputs.items():
            if "out" in k:
                semantic_loss += self.criterion(v, target)
            elif "aux" in k:
                semantic_aux += self.criterion(v, target)
        return semantic_loss, semantic_aux
