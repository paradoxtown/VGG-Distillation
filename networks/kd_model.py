import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import networks.net as net
from utils.utils import load_t_model, print_model_parm_nums, load_s_model
from utils.criterion import CriterionForDistribution, CriterionHT


class NetModel:
    @staticmethod
    def name():
        return 'kd_class'

    def __init__(self, args):
        cudnn.enabled = True
        self.args = args
        # device = args.device

        student = net.SimpleNet(args.num_classes)
        if args.load_student:
            load_s_model(student, args.s_ckpt_path)
        print_model_parm_nums(student, 'student_model')
        self.student = student.cuda()

        teacher = net.VGGNet(args.num_classes)
        load_t_model(teacher, args.t_ckpt_path)
        print_model_parm_nums(teacher, 'teacher_model')
        self.teacher = teacher.cuda()

        # todo
        # self.optimizer = optim.SGD({'params': filter(lambda p: p.requires_grad, self.student.parameters()),
        #                             'initial_lr': args.lr}, args.lr, momentum=args.momentum,
        #                             weight_decay=args.weight_decay)

        # regressor_params = list(map(id, self.student.regressor.parameters()))
        # base_params = filter(lambda p: id(p) not in regressor_params, self.student.parameters())
        # self.optimizer = optim.SGD(base_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer = optim.SGD(self.student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # self.optimizer = optim.RMSprop(self.student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        # criterion cross entropy + soft-target distribution align + HT
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_for_distribution = CriterionForDistribution()
        self.criterion_ht = CriterionHT()

        self.loss_ce = 0.0
        self.loss_it = 0.0
        self.loss_ht = 0.0

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
        if args.ce:
            l_ce = args.lambda_ce * self.criterion_ce(self.preds_s[2], self.labels)
            self.loss_ce = l_ce.item()
            loss += l_ce
        if args.it:
            l_it = args.lambda_it * self.criterion_for_distribution(self.preds_s[1], self.preds_t[1])
            self.loss_it = l_it.item()
            loss += l_it
        if args.ht:
            l_ht = args.lambda_ht * self.criterion_ht(self.preds_s[0], self.preds_t[0])
            self.loss_ht = l_ht.item()
            loss += l_ht
        loss.backward()
        self.loss = loss.item()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.student_backward()
        self.optimizer.step()

    def evaluate_model(self, total, correct):
        _, predicted = self.preds_s[2].max(1)
        self.total = total + self.labels.size(0)
        self.correct = correct + predicted.eq(self.labels).sum().item()
        self.acc = self.correct / self.total
        return self.total, self.correct

    def print_info(self, epoch, step):
        if self.args.it:
            print('[%2d, %5d] loss: %.3f, loss_ce: %.3f, loss_it: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, self.loss_it, 100.*self.acc))
        elif self.args.ce:
            print('[%d, %5d] loss: %.3f, loss_ce: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, 100.*self.acc))
        else:
            print('[%d, %5d] loss: %.3f, loss_ht: %.3f' %
                  (epoch + 1, step + 1, self.loss, self.loss_ht))

    def save_ckpt(self, time, epoch):
        ckpt_path = ''
        if not self.args.ht:
            # ckpt_path = './checkpoint/distill/ckpt_{}_{}_{}.pth'.format(time, epoch+1, self.acc)
            ckpt_path = './checkpoint/distill/ckpt_{}_it.pth'.format(epoch+1)
            torch.save(self.student.state_dict(), ckpt_path)
        else:
            # ckpt_path = './checkpoint/distill/ckpt_{}_{}.pth'.format(time, epoch+1)
            ckpt_path = './checkpoint/distill/ckpt_{}_ht.pth'.format(epoch+1)
            torch.save(self.student.state_dict(), ckpt_path)
        return ckpt_path
