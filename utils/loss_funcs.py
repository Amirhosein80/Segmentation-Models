import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Optional


class BootstrappedCE(nn.Module):
    def __init__(self, loss_th: float = 0.3, ignore_index: int = 255, label_smoothing: float = 0.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.threshold = loss_th
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none", label_smoothing=label_smoothing, weight=weight
        )

    def forward(self, output, labels):
        pixel_losses = self.criterion(output, labels).contiguous().view(-1)
        k = torch.numel(labels) // 16
        mask = (pixel_losses > self.threshold)
        if torch.sum(mask).item() > k:
            pixel_losses = pixel_losses[mask]
        else:
            pixel_losses, _ = torch.topk(pixel_losses, k)
        return pixel_losses.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255, use_iou_loss: bool = True):
        super().__init__()
        self.use_iou_loss = use_iou_loss
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        calculate loss :)
        :param output: output of nural network shape: (B, C, H, W)
        :param label: true labels shape: (B, H, W)
        :return: loss value
        """
        _, c, _, _ = output.shape

        label = label.reshape(-1)
        not_ignored = label != self.ignore_index
        label = label[not_ignored]
        label = f.one_hot(label, num_classes=c).type(output.type())

        output = output.permute(0, 2, 3, 1)
        output = output.reshape(-1, c)
        output = output[not_ignored]

        iou_loss = 0
        if self.use_iou_loss:
            intersect = output * label
            union = output + label - intersect + 1e-8
            iou_loss = 1.0 - (torch.sum(intersect) / torch.sum(union))
            iou_loss = iou_loss ** (1 / self.gamma)

        p = torch.sigmoid(output)
        ce_loss = f.binary_cross_entropy_with_logits(output, label, reduction="none")
        p_t = p * label + (1 - p) * (1 - label)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)
            loss = alpha_t * loss
        return loss.mean() + iou_loss


if __name__ == "__main__":
    n = torch.randint(0, 18, (8, 512, 512))
    o = torch.ones(8, 19, 512, 512)
    n[1] = torch.ones(512, 512) * 255
    print(n.shape)
    fl = FocalLoss()
    a = fl(o, n)
    print(a.shape)
    print(n.shape)
    print(o.shape)
