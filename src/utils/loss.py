import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                focal_loss = self.alpha * focal_loss
            else:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        LDAM Loss for imbalanced datasets.
        Args:
            cls_num_list (list): Number of samples per class [n_class_1, n_class_2, ...].
            max_m (float): Maximum margin (default: 0.5).
            weight (torch.Tensor, optional): Per-class weights for reweighting (default: None).
            s (float): Scaling factor for logits (default: 30).
        """
        super(LDAMLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.max_m = max_m
        self.s = s
        self.weight = weight

        # Compute margins based on class frequencies
        m_list = 1.0 / torch.sqrt(
            torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float))
        )
        m_list = m_list * (max_m / torch.max(m_list))
        self.m_list = m_list.cuda() if torch.cuda.is_available() else m_list

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (pre-softmax), shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth labels, shape [batch_size].
        Returns:
            loss (torch.Tensor): LDAM loss.
        """
        batch_m = self.m_list[targets]  # Get margin for each sample's class
        batch_m = batch_m.view(-1, 1)  # Reshape: [batch_size] -> [batch_size, 1]

        # Shift logits for the target class by margin
        x_m = logits - batch_m * self.s  # Scale margin by s
        output = torch.where(
            torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0).bool(),
            x_m,
            logits,
        )

        # Compute softmax cross-entropy with modified logits
        loss = F.cross_entropy(
            self.s * output, targets, weight=self.weight, reduction="mean"
        )
        return loss
