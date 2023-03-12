import copy
import datetime
import os
from collections import OrderedDict

import tensorboardX as ts
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.optim as optim
import tqdm.autonotebook as tqdm

from loss_funcs import BootstrappedCE, FocalLoss
from models.model_utils import reparameterize_model
from utils.model_ema import ModelEma
from utils.train_utils import ConfusionMatrix, AverageMeter, EarlyStopping, set_seed, get_lr


class Trainer(object):
    """
    A Trainer Class for test & evaluate model :)
    """

    def __init__(self, model, config, name: str, device: str):
        """
        :param model:
        :param config:
        :param name:
        :param device:
        """
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
        self.writer = ts.SummaryWriter(f"runs/{self.name}_{self.now}")
        self.early_stopping = EarlyStopping(self.config)
        self._get_optimizer()
        self._get_criterion()
        self._resume()

    def _overfit(self, model, epoch, batch):
        """
        overfit test :)
        :param model: model
        :param epoch: current epoch
        :param batch: batch to overfit
        :return:
        """
        model.train()
        loss_total = AverageMeter()
        semantic_total = AverageMeter()
        aux_total = AverageMeter()
        metric = ConfusionMatrix(num_classes=self.config.NUM_CLASSES)
        set_seed(epoch)
        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            outputs = model(inputs)
            semantic_loss, semantic_aux = self._criterion_forward(outputs, targets)
            loss = semantic_loss + (0.4 * semantic_aux)
            metric.update(targets.flatten(), outputs["out"].argmax(dim=1).flatten())
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.model_ema is not None:
            self.model_ema.update(model)

        loss_total.update(loss)
        semantic_total.update(semantic_loss)
        aux_total.update(semantic_aux)
        torch.cuda.empty_cache()
        print(f"Train -> Epoch:{epoch} Loss:{loss_total.avg:.4}"
              f" Out Loss: {semantic_total.avg:.4} Aux Loss: {0.4 * aux_total.avg:.4},"
              f" LR: {get_lr(self.optimizer)}")

    def _train(self, model, epoch, dataloader):
        # set model for train
        model.train()

        # save losses of each step
        loss_total = AverageMeter()
        semantic_total = AverageMeter()
        aux_total = AverageMeter()

        # Confusion matrix to calculate iou
        metric = ConfusionMatrix(num_classes=self.config.NUM_CLASSES)
        set_seed(epoch - 1)
        loop = tqdm.tqdm(dataloader, total=len(dataloader))

        # one epoch loop
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):

                # forward step & calc loss & metric
                outputs = model(inputs)
                semantic_loss, semantic_aux = self._criterion_forward(outputs, targets)
                loss = semantic_loss + (0.4 * semantic_aux)

                # gradient accumulation
                if self.config.GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
                metric.update(targets.flatten(), outputs["out"].argmax(dim=1).flatten())

            # backward step
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:

                # if you use gradient accumulation i suggest you to use gradient norm  to prevent gradient exploding
                if self.config.GRADIENT_NORM is not None:
                    self.scaler.unscale_(self.optimizer)
                    if hasattr(model, "get_params"):
                        params = model.get_params(lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY)
                    else:
                        params = model.parameters()
                    nn.utils.clip_grad_norm_(params, self.config.GRADIENT_NORM)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.model_ema is not None:
                    self.model_ema.update(model)
                self.optimizer.zero_grad()

                loss_total.update(loss * self.config.GRADIENT_ACCUMULATION_STEPS)
                semantic_total.update(semantic_loss)
                aux_total.update(semantic_aux)

            torch.cuda.empty_cache()
            loop.set_description(f"Train -> Epoch:{epoch} Loss:{loss_total.avg:.4}"
                                 f" Out Loss: {semantic_total.avg:.4} Aux Loss: {0.4 * aux_total.avg:.4},"
                                 f" LR: {get_lr(self.optimizer)}")
        self.scheduler.step()
        miou = metric.calculate()

        state = {
            'model': model.state_dict(),
            'acc': miou,
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
        path = f"{self.name}" + '.pth'
        torch.save(state, path)

        return miou, semantic_total.avg.item()

    def _test(self, model, epoch, dataloader):
        # use Reparameterize model or Original model
        if epoch % 5 == 0:
            print("Reparameterize model")
            rmodel = reparameterize_model(model)
        else:
            print("Original model")
            rmodel = copy.deepcopy(model)

        # set model for train
        rmodel.eval()

        # save losses of each step
        loss_total = AverageMeter()
        semantic_total = AverageMeter()
        aux_total = AverageMeter()

        # Confusion matrix to calculate iou
        metric = ConfusionMatrix(num_classes=self.config.NUM_CLASSES)
        set_seed(epoch - 1)
        loop = tqdm.tqdm(dataloader, total=len(dataloader))

        # one epoch loop
        for batch_idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                outputs = rmodel(inputs)
                semantic_loss, semantic_aux = self._criterion_forward(outputs, targets)
                loss = semantic_loss + (0.4 * semantic_aux)
                metric.update(targets.flatten(), outputs["out"].argmax(dim=1).flatten())
            loss_total.update(loss)
            semantic_total.update(semantic_loss)
            aux_total.update(semantic_aux)
            torch.cuda.empty_cache()
            loop.set_description(f"Valid -> Epoch:{epoch} Loss:{loss_total.avg:.4}"
                                 f" Out Loss: {semantic_total.avg:.4} Aux Loss: {0.4 * aux_total.avg:.4}")
        miou = metric.calculate()

        return miou, semantic_total.avg.item()

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
        self.start_epoch, self.best_acc = 0, 0.0
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
            params = self.model.get_params(lr=self.config.LR,
                                           weight_decay=self.config.WEIGHT_DECAY if not self.config.OVERFIT_TEST else 0)
        else:
            params = self.model.parameters()

        if self.config.OPTIMIZER == "ADAMW":
            self.optimizer = optim.AdamW(params, lr=self.config.LR,
                                         weight_decay=self.config.WEIGHT_DECAY if not self.config.OVERFIT_TEST else 0,
                                         betas=self.config.ADAMW_BETAS)

        elif self.config.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(params, lr=self.config.LR,
                                       weight_decay=self.config.WEIGHT_DECAY if not self.config.OVERFIT_TEST else 0,
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

        if self.config.WARMUP_EPOCHS > 0 and not self.config.OVERFIT_TEST:
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

    def fit(self, trainloader, testloader):

        for epoch in range(self.start_epoch + 1, self.config.EPOCHS + 1):
            if epoch % 5 == 0:
                model_type = "Reparameterize"
            else:
                model_type = "Original"
            train_acc, train_loss = self._train(model=self.model, epoch=epoch, dataloader=trainloader)
            test_acc, test_loss = self._test(model=self.model, epoch=epoch, dataloader=testloader)
            self.writer.add_scalar('Loss/train', train_loss.item(), epoch)
            self.writer.add_scalar('Metric/train', train_acc, epoch)
            self.writer.add_scalar('LR/train', get_lr(self.optimizer), epoch)
            self.writer.add_scalar('Loss/valid', test_loss.item(), epoch)
            self.writer.add_scalar('Metric/valid', test_acc, epoch)
            if self.model_ema is not None:
                ema_acc, ema_loss = self._test(model=self.model_ema.module, epoch=epoch, dataloader=testloader)
                self.writer.add_scalar('Loss/valid_ema', ema_loss.item(), epoch)
                self.writer.add_scalar('Metric/valid_ema', ema_acc, epoch)
            else:
                ema_acc, ema_loss = 0, 0
            self.writer.add_hparams(hparam_dict={
                "lr": self.config.LR,
                "weight_decay": self.config.WEIGHT_DECAY,
                "optimizer": self.config.OPTIMIZER,
                "batch_size": self.config.BATCH_SIZE,
                "loss": self.config.LOSS,
                "grad_accum": self.config.GRADIENT_ACCUMULATION_STEPS,
                "grad_norm": self.config.GRADIENT_NORM,
                },
                metric_dict={
                    "loss": test_loss,
                    "acc": test_acc,
            })

            self._save(acc=test_acc, epoch=epoch)

            self.early_stopping(train_loss=train_loss, validation_loss=test_loss)
            if self.early_stopping.early_stop:
                print(f"Early Stop at Epoch: {epoch}")
                break

            with open(f"{self.name}.txt", "a") as f:
                f.write(f"Epoch: {epoch}, Model: {model_type},"
                        f" Train mIOU: {train_acc}, Train loss: {train_loss},"
                        f" Valid mIOU: {test_acc}, Valid loss: {test_loss},"
                        f" EMA mIOU: {ema_acc}, EMA loss: {ema_loss}\n")

            print()
