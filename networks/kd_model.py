import utils.parallel as parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch
import networks.net as net
import networks.gan as gan
from utils.utils import *
from utils.criterion import *


class NetModel:
    @staticmethod
    def name():
        return 'kd_seg'

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

        student = net.SimpleNet()
        load_s_model(args, student, False)
        print_model_parm_nums(student, 'student_model')
        self.parallel_student = self.data_parallel_model_process(student, 'train', device)
        self.student = student

        net_arch16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5', "FC1",
                      "FC2", "FC"]
        teacher = net.VGGNet(net_arch16, args.num_classes)
        load_t_model(teacher, args.T_ckpt_path)
        print_model_parm_nums(teacher, 'teacher_model')
        self.parallel_teacher = self.data_parallel_model_process(teacher, 'eval', device)
        self.teacher = teacher

        d_model = gan.Discriminator()
        load_d_model(args, d_model, False)
        print_model_parm_nums(d_model, 'd_model')
        self.parallel_d = self.data_parallel_model_process(d_model, 'train', device)

        # todo
        self.G_solver = optim.SGD({'params': filter(lambda p: p.requires_grad, self.student.parameters()),
                                   'initial_lr': args.lr_g},
                                  args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)
        self.D_solver = optim.SGD({'params': filter(lambda p: p.requires_grad, d_model.parameters()),
                                   'initial_lr': args.lr_d},
                                  args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

        self.best_mean_IU = args.best_mean_IU

        # criterion cross entropy
        self.criterion = self.data_parallel_criterion_process(CriterionDSN())
        self.criterion_pixel_wise = self.data_parallel_criterion_process(CriterionPixelWise())
        self.criterion_pair_wise = self.data_parallel_criterion_process(CriterionPairWiseForWholeFeatAfterPool())
        self.criterion_adv = self.data_parallel_criterion_process(CriterionAdv())
        if args.adv_loss_type == 'wgan-gp':
            self.criterion_AdditionalGP = self.data_parallel_criterion_process(CriterionAdditionalGP())
        self.criterion_adv_for_G = self.data_parallel_criterion_process(CriterionAdvForG())

        self.mc_G_loss = 0.0
        self.pi_G_loss = 0.0
        self.pa_G_loss = 0.0
        self.D_loss = 0.0

        cudnn.benchmark = True
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        self.images = None
        self.labels = None
        self.preds_t = None
        self.preds_s = None

    def set_input(self, data):
        # args = self.args
        images, labels, _, _ = data
        self.images = images.cuda()
        self.labels = labels.long().cuda()

    @staticmethod
    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** power)

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
        pass

    def discriminator_backward(self):
        pass

    def optimize_parameters(self):
        pass

    def evaluate_model(self):
        pass

    def print_info(self):
        pass

    def save_ckpt(self):
        pass