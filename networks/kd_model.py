import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import networks.net as net
from utils.utils import load_t_model, print_model_parm_nums
from utils.criterion import CriterionForDistribution, CriterionPixelWise


class NetModel:
    @staticmethod
    def name():
        return 'kd_class'

    def __init__(self, args):
        cudnn.enabled = True
        self.args = args
        # device = args.device

        student_arch = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
        student = net.SimpleNet(student_arch, args.num_classes)
        # load_s_model(args, student, False)
        print_model_parm_nums(student, 'student_model')
        self.student = student.cuda()

        teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
        teacher = net.VGGNet(teacher_arch, args.num_classes)
        load_t_model(teacher, args.t_ckpt_path)
        print_model_parm_nums(teacher, 'teacher_model')
        self.teacher = teacher.cuda()

        # todo
        # self.optimizer = optim.SGD({'params': filter(lambda p: p.requires_grad, self.student.parameters()),
        #                             'initial_lr': args.lr}, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer = optim.SGD(self.student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # criterion cross entropy + soft-target distribution align + pixel-wise
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_up = nn.MSELoss()
        self.criterion_for_distribution = CriterionForDistribution()
        self.criterion_pixel_wise = CriterionPixelWise()

        self.loss_ce = 0.0
        self.loss_up = 0.0
        self.loss_pi = 0.0
        self.loss_it = 0.0

        self.total = 0
        self.correct = 0
        self.acc = 0.0
        self.images = None
        self.labels = None
        self.preds_t = None
        self.preds_s = None
        self.loss = None

    def set_input(self, dat):
        images, labels = dat
        self.images = images.cuda()
        self.labels = labels.cuda()

    # @staticmethod
    # def lr_poly(base_lr, iteration, max_iter, power):
    #     return base_lr * ((1 - float(iteration) / max_iter) ** power)

    # def adjust_learning_rate(self, base_lr, optimizer, i_iter):
    #     args = self.args
    #     lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
    #     optimizer.param_groups[0]['lr'] = lr
    #     return lr

    def forward(self):
        with torch.no_grad():
            self.preds_t = self.teacher.eval()(self.images)
        self.preds_s = self.student.train()(self.images)

    def student_backward(self):
        args = self.args
        loss = 0.0
        l_ce = 0.1 * self.criterion_ce(self.preds_s[1], self.labels)
        self.loss_ce = l_ce.item()
        loss += l_ce
        l_up = self.criterion_up(self.preds_s[2], self.images)
        self.loss_up = l_up.item()
        loss += l_up
        if args.pi:
            l_pi = args.lambda_pi * self.criterion_pixel_wise(self.preds_s[2], self.preds_t[2])
            self.loss_pi = l_pi.item()
            loss += l_pi
        if args.it:
            l_it = args.lambda_it * self.criterion_for_distribution(self.preds_s[1], self.preds_t[1])
            self.loss_it = l_it.item()
            loss += l_it
        loss.backward()
        self.loss = loss.item()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.student_backward()
        self.optimizer.step()

    def evaluate_model(self, total, correct):
        _, predicted = self.preds_s[1].max(1)
        self.total = total + self.labels.size(0)
        self.correct = correct + predicted.eq(self.labels).sum().item()
        self.acc = self.correct / self.total
        return self.total, self.correct

    def print_info(self, epoch, step):
        print('[%d, %5d] loss: %.3f, loss_ce: %.3f, loss_up: %.3f, acc: %.3f%%' %
              (epoch + 1, step + 1, self.loss, self.loss_ce, self.loss_up, 100.*self.acc))

    def save_ckpt(self, time, epoch):
        torch.save(self.student.state_dict(), './checkpoint/distill/ckpt_{}_{}_{}.pth'.format(time, epoch, self.acc))
