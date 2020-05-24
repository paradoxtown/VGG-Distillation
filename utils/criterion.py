import torch.nn as nn
import torch
import torch.nn.functional as F


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


class CriterionSoftTarget(nn.Module):
    def __init__(self):
        super(CriterionSoftTarget, self).__init__()
        self.T = 4.0

    def forward(self, preds_s, preds_t):
        loss = F.kl_div(F.log_softmax(preds_s/self.T, dim=1), F.softmax(preds_t/self.T, dim=1), reduction='batchmean') * self.T * self.T
        return loss


class CriterionFSP(nn.Module):
    def __init__(self):
        super(CriterionFSP, self).__init__()

    def forward(self, preds_s, preds_t):
        pass


class CriterionHT(nn.Module):
    def __init__(self):
        super(CriterionHT, self).__init__()
        self.l2 = nn.MSELoss()

    def forward(self, guided_ws, hint_ws):
        batch_s = guided_ws.size(0)
        batch_t = hint_ws.size(0)
        assert batch_s == batch_t, 'the batch size of student is not consistent with teacher'
        steps = batch_s
        loss = 0.0
        for s in range(steps):
            loss += 0.5 * self.l2(guided_ws[s], hint_ws[s])
        loss /= steps
        return loss


class CriterionLogits(nn.Module):
    def __init__(self):
        super(CriterionLogits, self).__init__()

    def forward(self, preds_s, preds_t):
        loss = F.mse_loss(preds_s, preds_t)
        return loss


class CriterionSP(nn.Module):
    def __init__(self):
        super(CriterionSP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        g_s = torch.mm(fm_s, fm_s.t())
        norm_g_s = F.normalize(g_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        g_t = torch.mm(fm_t, fm_t.t())
        norm_g_t = F.normalize(g_t, p=2, dim=1)

        loss = F.mse_loss(norm_g_s, norm_g_t)

        return loss


class CriterionAT(nn.Module):
    def __init__(self):
        super(CriterionAT, self).__init__()
        self.p = 2.0

    def forward(self, fm_s, fm_t):
        loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))
        return loss

    def attention_map(self, fm, eps=1e-6):
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am
