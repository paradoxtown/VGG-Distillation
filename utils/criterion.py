import torch.nn as nn
import torch


class CriterionDSN(nn.Module):
    """
    loss multi-classes
    l_mc = cross_entropy
    """

    def __init__(self):
        super(CriterionDSN, self).__init__()

    def forward(self, preds_s, preds_t):
        pass


class CriterionPixelWise(nn.Module):
    """
    loss pixel wise
    l_pi = mse_loss
    """

    def __init__(self):
        super(CriterionPixelWise, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, preds_s, preds_t):
        loss = self.criterion(preds_s, preds_t)
        return loss


class CriterionForDistribution(nn.Module):
    """
    loss inter distribution
    l_inter = - sum ^N _{p=1} y_p log(hat{y}_p)
    """

    def __init__(self):
        super(CriterionForDistribution, self).__init__()

    def forward(self, preds_s, preds_t):
        batch_s = preds_s.size(0)
        batch_t = preds_t.size(0)
        assert batch_s == batch_t, 'the batch size of student is not insistance with teacher'
        steps = batch_s
        classes = preds_s.size(1)
        loss = 0.0
        for s in range(steps):
            for c in range(classes):
                loss += preds_t[s][c] * torch.log(preds_s[s][c])
        loss /= steps
        loss = - loss
        return loss
