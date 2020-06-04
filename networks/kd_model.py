import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import networks.net as net
from utils.utils import load_t_model, print_model_parm_nums, load_s_model
from utils.criterion import CriterionForDistribution, CriterionHT, \
    CriterionSoftTarget, CriterionLogits, CriterionSP, CriterionAT


class NetModel:
    @staticmethod
    def name():
        return 'kd_class'

    def __init__(self, args):
        cudnn.enabled = True
        self.args = args
        # device = args.device

        student = net.SimpleNet16A(args.num_classes)
        if args.load_student:
            load_s_model(student, args.s_ckpt_path)
        print_model_parm_nums(student, 'student_model')
        self.student = student.cuda()

        teacher = net.VGGNet16s(args.num_classes)
        load_t_model(teacher, args.t_ckpt_path)
        print_model_parm_nums(teacher, 'teacher_model')
        self.teacher = teacher.cuda()

        self.optimizer = optim.SGD(self.student.parameters(), lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay, nesterov=True)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_for_distribution = CriterionForDistribution()
        self.criterion_st = CriterionSoftTarget()
        self.criterion_ht = CriterionHT()
        self.criterion_lg = CriterionLogits()
        self.criterion_sp = CriterionSP()
        self.criterion_at = CriterionAT()

        self.loss_ce = 0.0
        self.loss_it = 0.0
        self.loss_st = 0.0
        self.loss_lg = 0.0
        self.loss_sp = 0.0
        self.loss_at = 0.0
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

    def adjust_learning_rate(self, optimizer, epoch):
        #     args = self.args
        #     lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        #     optimizer.param_groups[0]['lr'] = lr
        #     return lr
        scale = 0.1
        lr_list = [self.args.lr] * 100
        lr_list += [self.args.lr * scale] * 50
        lr_list += [self.args.lr * scale * scale] * 50

        lr = lr_list[epoch - 1]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def forward(self):
        with torch.no_grad():
            self.preds_t = self.teacher.eval()(self.images)
        self.preds_s = self.student.train()(self.images)

    def student_backward(self):
        args = self.args
        loss = 0.0
        if args.ce:
            l_ce = args.lambda_ce * self.criterion_ce(self.preds_s[-2], self.labels)
            self.loss_ce = l_ce.item()
            loss = loss + l_ce
        if args.it:
            l_it = args.lambda_it * self.criterion_for_distribution(self.preds_s[1], self.preds_t[1])
            self.loss_it = l_it.item()
            loss = loss + l_it
        if args.st:
            l_st = args.lambda_st * self.criterion_st(self.preds_s[-2], self.preds_t[-2])
            self.loss_st = l_st.item()
            loss = loss + l_st
        if args.lg:
            l_lg = args.lambda_lg * self.criterion_lg(self.preds_s[-2], self.preds_t[-2])
            self.loss_lg = l_lg.item()
            loss = loss + l_lg
        if args.sp:
            # l_sp = args.lambda_sp * (self.criterion_sp(self.preds_s[3], self.preds_t[3]) + self.criterion_sp(self.preds_s[2], self.preds_t[2])) / 2.0
            l_sp = args.lambda_sp * self.criterion_sp(self.preds_s[3], self.preds_t[3])
            self.loss_sp = l_sp.item()
            loss = loss + l_sp
        if args.at:
            l_at = args.lambda_at * self.criterion_at(self.preds_s[2], self.preds_t[2])
            self.loss_at = l_at.item()
            loss = loss + l_at
        if args.ht:
            l_ht = args.lambda_ht * self.criterion_ht(self.preds_s[5], self.preds_t[5])
            self.loss_ht = l_ht.item()
            loss = loss + l_ht
        loss.backward()
        self.loss = loss.item()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.student_backward()
        self.optimizer.step()

    def evaluate_model(self, total, correct):
        _, predicted = self.preds_s[-2].max(1)
        self.total = total + self.labels.size(0)
        self.correct = correct + predicted.eq(self.labels).sum().item()
        self.acc = self.correct / self.total
        return self.total, self.correct

    def print_info(self, epoch, step):
        if self.args.it:
            print('[%2d, %5d] loss: %.3f, loss_ce: %.3f, loss_it: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, self.loss_it, 100. * self.acc))
        elif self.args.st:
            print('[%2d, %5d] loss: %.3f, loss_ce: %.3f, loss_st: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, self.loss_st, 100. * self.acc))
        elif self.args.lg:
            print('[%2d, %5d] loss: %.3f, loss_ce: %.3f, loss_lg: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, self.loss_lg, 100. * self.acc))
        elif self.args.sp:
            print('[%2d, %5d] loss: %.3f, loss_ce: %.3f, loss_sp: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, self.loss_sp, 100. * self.acc))
        elif self.args.at:
            print('[%2d, %5d] loss: %.3f, loss_ce: %.3f, loss_at: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, self.loss_at, 100. * self.acc))
        elif self.args.ce:
            print('[%d, %5d] loss: %.3f, loss_ce: %.3f, acc: %.3f%%' %
                  (epoch + 1, step + 1, self.loss, self.loss_ce, 100. * self.acc))
        else:
            print('[%d, %5d] loss: %.3f, loss_ht: %.3f' %
                  (epoch + 1, step + 1, self.loss, self.loss_ht))

    def save_ckpt(self, time, epoch):
        ckpt_path = ''
        if self.args.st:
            # ckpt_path = './checkpoint/distill/ckpt_{}_{}_{}.pth'.format(time, epoch+1, self.acc)
            ckpt_path = './checkpoint/distill/st/ckpt_{}_st.pth'.format(epoch + 1)
        elif self.args.lg:
            ckpt_path = './checkpoint/distill/lg/ckpt_{}_lg.pth'.format(epoch + 1)
        elif self.args.sp:
            ckpt_path = './checkpoint/distill/sp/ckpt_{}_sp.pth'.format(epoch + 1)
        elif self.args.at:
            ckpt_path = './checkpoint/distill/at/ckpt_{}_at.pth'.format(epoch + 1)
        elif self.args.it:
            ckpt_path = './checkpoint/distill/it/ckpt_{}_it.pth'.format(epoch + 1)
        elif self.args.ht:
            # ckpt_path = './checkpoint/distill/ckpt_{}_{}.pth'.format(time, epoch+1)
            ckpt_path = './checkpoint/distill/ht/ckpt_{}_ht.pth'.format(epoch + 1)
        torch.save(self.student.state_dict(), ckpt_path)
        return ckpt_path
