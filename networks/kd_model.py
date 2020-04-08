import utils.parallel as parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import networks.net as net
# from utils.utils import *
from utils.criterion import *


class NetModel:
    @staticmethod
    def name():
        return 'kd_class'

    @staticmethod
    def data_parallel_model_process(model, is_eval='train', device='cuda'):
        parallel_model = parallel.DataParallelModel(model)
        if is_eval == 'eval':
            parallel_model.eval()
        elif is_eval == 'train':
            parallel_model.train()
        else:
            raise ValueError('is_eval should be eval or train')
        parallel_model.float()
        parallel_model.to(device)
        return parallel_model

    @staticmethod
    def data_parallel_criterion_process(criterion):
        criterion = parallel.DataParallelCriterion(criterion)
        criterion.cuda()
        return criterion

    def __init__(self, args):
        cudnn.enabled = True
        self.args = args
        device = args.device

        student_arch = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
        student = net.SimpleNet(student_arch, args.num_classes)
        load_s_model(args, student, False)
        print_model_parm_nums(student, 'student_model')
        self.parallel_student = self.data_parallel_model_process(student, 'train', device)
        self.student = student

        teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
        teacher = net.VGGNet(teacher_arch, args.num_classes)
        load_t_model(teacher, args.T_ckpt_path)
        print_model_parm_nums(teacher, 'teacher_model')
        self.parallel_teacher = self.data_parallel_model_process(teacher, 'eval', device)
        self.teacher = teacher

        # todo
        self.G_solver = optim.SGD({'params': filter(lambda p: p.requires_grad, self.student.parameters()),
                                   'initial_lr': args.lr_g},
                                  args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)

        # criterion cross entropy + soft-target distribution + pair-wise + pixel-wise
        self.criterion = self.data_parallel_criterion_process(nn.CrossEntropyLoss())
        self.criterion_for_distribution = self.data_parallel_criterion_process(CriterionForDistribution())
        self.criterion_pixel_wise = self.data_parallel_criterion_process(CriterionPixelWise())
        self.criterion_pair_wise = \
            self.data_parallel_criterion_process(CriterionPairWiseForWholeFeatAfterPool(args.scale, -1))

        self.mc_G_loss = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0
        self.ho_G_loss = 0.0

        cudnn.benchmark = True
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        self.images = None
        self.labels = None
        self.preds_t = None
        self.preds_s = None
        self.g_loss = None

    def set_input(self, data):
        images, labels, _, _ = data
        self.images = images.cuda()
        self.labels = labels.long().cuda()

    @staticmethod
    def lr_poly(base_lr, iteration, max_iter, power):
        return base_lr * ((1 - float(iteration) / max_iter) ** power)

    def adjust_learning_rate(self, base_lr, optimizer, i_iter):
        args = self.args
        lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        optimizer.param_groups[0]['lr'] = lr
        return lr

    def forward(self):
        args = self.args
        with torch.no_grad():
            self.preds_t = self.parallel_teacher.eval()(self.images, parallel=args.parallel)
        self.preds_s = self.parallel_student.train()(self.images, parallel=args.parallel)

    def student_backward(self):
        args = self.args
        g_loss = 0.0
        temp = self.criterion(self.preds_s, self.labels, is_target_scattered=False)
        # temp_t = self.criterion(self.preds_t, self.labels, is_target_scattered=False)
        self.mc_G_loss = temp.item()
        g_loss += temp
        if args.pi:
            temp = args.lambda_pi * self.criterion_pixel_wise(self.preds_s, self.preds_t, is_target_scattered=True)
            self.pi_G_loss = temp.item()
            g_loss += temp
        if args.pa:
            # todo
            temp1 = self.criterion_pair_wise(self.preds_s, self.preds_t, is_target_scattered=True)
            self.pa_G_loss = temp1.item()
            g_loss = g_loss + args.labda_pa * temp1
        if args.ho:
            temp2 = self.criterion_for_distribution(self.preds_s, self.preds_t)
            self.ho_G_loss = temp2.item()
            g_loss = g_loss + args.labda_ho * temp2
        g_loss.backward()
        self.g_loss = g_loss.item()

    def optimize_parameters(self):
        self.forward()
        self.G_solver.zero_grad()
        self.student_backward()
        self.G_solver.step()

    def evaluate_model(self):
        pass

    def print_info(self):
        pass

    def save_ckpt(self):
        pass
