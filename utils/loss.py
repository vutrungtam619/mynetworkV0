import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, alpha=2.0, gamma=2.0, cls_w=1.0, reg_w=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cls_w = cls_w
        self.reg_w = reg_w

    def focal_loss(self, pred_heatmap, gt_heatmap):
        pred = torch.sigmoid(pred_heatmap).clamp(1e-6, 1.0 - 1e-6)
        pos_mask = (gt_heatmap == 1).float()
        neg_mask = (gt_heatmap < 1).float()

        pos_loss = -torch.log(pred) * ((1.0 - pred) ** self.alpha) * pos_mask
        neg_weight = (1.0 - gt_heatmap) ** self.gamma
        neg_loss = -torch.log(1.0 - pred) * (pred ** self.alpha) * neg_weight * neg_mask
        num_pos = pos_mask.sum()
        if num_pos < 1:
            return neg_loss.sum()

        return (pos_loss.sum() + neg_loss.sum()) / num_pos

    def reg_loss(self, pred_offset, gt_offsetmap, offset_mask):
        if offset_mask.sum() == 0:
            return pred_offset.sum() * 0.0
        mask = offset_mask.expand_as(pred_offset)  # (B, 2, H, W)

        return F.smooth_l1_loss(
            pred_offset[mask],
            gt_offsetmap[mask],
            reduction="mean",
            beta=1.0
        )

    def forward(self, pred_heatmap, pred_offsetmap, gt_heatmap, gt_offsetmap, gt_offsetmask):
        heat_loss = self.focal_loss(pred_heatmap, gt_heatmap)
        offset_loss = self.reg_loss(pred_offsetmap, gt_offsetmap, gt_offsetmask)
        total_loss = self.cls_w * heat_loss + self.reg_w * offset_loss
        return {'focal_loss': heat_loss, 'reg_loss': offset_loss, 'total_loss': total_loss}
