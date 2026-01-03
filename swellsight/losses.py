import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, w_height=1.0, w_wave_type=1.0, w_direction=1.0):
        super().__init__()
        self.w_height = w_height
        self.w_wave_type = w_wave_type
        self.w_direction = w_direction

        self.reg = nn.SmoothL1Loss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, pred_h, pred_wt, pred_dir, y_h, y_wt, y_dir, sample_w):
        w = sample_w.view(-1)

        loss_h = self.reg(pred_h.view(-1), y_h.view(-1))
        loss_h = (loss_h * w).mean()

        loss_wt = self.ce(pred_wt, y_wt.view(-1))
        loss_wt = (loss_wt * w).mean()

        loss_dir = self.ce(pred_dir, y_dir.view(-1))
        loss_dir = (loss_dir * w).mean()

        total = self.w_height * loss_h + self.w_wave_type * loss_wt + self.w_direction * loss_dir
        return total, {"loss_h": float(loss_h.item()), "loss_wt": float(loss_wt.item()), "loss_dir": float(loss_dir.item())}
