import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 主旨是简单，所以maybe loss不需要太复杂
# 但是有时侯，搞出来一个很精致的loss，其实也是很强的
# loss的设计，要看你的目的是什么，度量学习有点重要貌似

# l1 loss  or  mse loss


# l2 loss



# NCC loss or CC loss



# SIMM loss




# cosine 相似度 loss






# 一些分割的loss

# dice loss
class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
    def forward(self, probs, targets):
        # probs = torch.zeros([3,250])
        # targets = torch.zeros([3,250])
        # probs[:,175:] = 1
        # targets[:,125:] = 1
        num = targets.size(0)
        smooth = 1

        m1 = probs
        m2 = targets
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score






















































