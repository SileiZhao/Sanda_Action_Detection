import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): 类别权重，用于平衡正负样本。
            gamma (float): 调节因子，用于减少易分类样本的权重。
            reduction (str): 损失计算方式，支持 'mean'、'sum' 或 'none'。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算概率
        pt = torch.exp(-ce_loss)

        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 根据 reduction 参数返回损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss