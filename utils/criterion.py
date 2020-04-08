from torch.nn import functional as F
from utils.utils import *
from torch.autograd import Variable
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
    l_pi(S) = KL / (W * H)
    """

    def __init__(self):
        super(CriterionPixelWise, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, preds_s, preds_t):
        preds_t[0].detach()
        assert preds_s[0].shape == preds_t[0].shape, 'the dimension of teacher is not consist of students'
        N, C, W, H = preds_s[0].shape
        softmax_pred_t = F.softmax(preds_t[0].permute(0, 2, 3, 1).contiguous().view(-1, C), dim=1)
        log_softmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum(- softmax_pred_t * log_softmax(preds_s[0].permute(0, 2, 3, 1).contiguous().view(-1, C))))
        loss /= (W * H)
        return loss


class CriterionPairWiseForWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_index):
        super(CriterionPairWiseForWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale
        self.feat_index = feat_index

    def forward(self, preds_s, preds_t):
        feat_s = preds_s[self.feat_index]
        feat_t = preds_t[self.feat_index]
        feat_t.detach()

        total_w, total_h = feat_t.shape[2], feat_t.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        max_pool = nn.MaxPool2d(kernel_size=(patch_w, patch_w), stride=(patch_w, patch_h), padding=0, ceil_mode=True)
        loss = self.criterion(max_pool(feat_s), max_pool(feat_t))
        return loss


class CriterionForDistribution(nn.Module):
    def __init__(self):
        super(CriterionForDistribution, self).__init__()

    def forward(self, preds_s, preds_t):
        pass


class CriterionAdv(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdv, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_s, d_out_t):
        assert d_out_s[0].shape == d_out_t[0].shape, 'the output dim of D with teacher and student as input differ'
        '''teacher output'''
        d_out_real = d_out_t[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(d_out_real)
        elif self.adv_loss == 'hinge':
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')

        # apply Gumbel Softmax
        '''student output'''
        d_out_fake = d_out_s[0]
        if self.adv_loss == 'wgan-gp':
            d_loss_fake = d_out_fake.mean()
        elif self.adv_loss == 'hinge':
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return d_loss_real + d_loss_fake


class CriterionAdditionalGP(nn.Module):
    def __init__(self, d_net, lambda_gp):
        super(CriterionAdditionalGP, self).__init__()
        self.d = d_net
        self.lambda_gp = lambda_gp

    def forward(self, d_in_s, d_in_t):
        assert d_in_s[0].shape == d_in_t[0].shape, 'the output dim of D with teacher and student as input differ'

        real_images = d_in_t[0]
        fake_images = d_in_s[0]
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        out = self.D(interpolated)
        grad = torch.autograd.grad(outputs=out[0],
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(out[0].size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        d_loss = self.lambda_gp * d_loss_gp
        return d_loss


class CriterionAdvForG(nn.Module):
    def __init__(self, adv_type):
        super(CriterionAdvForG, self).__init__()
        if (adv_type != 'wgan-gp') and (adv_type != 'hinge'):
            raise ValueError('adv_type should be wgan-gp or hinge')
        self.adv_loss = adv_type

    def forward(self, d_out_s):
        g_out_fake = d_out_s[0]
        if self.adv_loss == 'wgan-gp':
            g_loss_fake = - g_out_fake.mean()
        elif self.adv_loss == 'hinge':
            g_loss_fake = - g_out_fake.mean()
        else:
            raise ValueError('args.adv_loss should be wgan-gp or hinge')
        return g_loss_fake
