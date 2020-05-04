import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config:
    @staticmethod
    def initialize():
        parser = argparse.ArgumentParser(description='knowledge-distillation')
        parser.add_argument('--num_classes', default=10, type=int)
        parser.add_argument('--t_ckpt_path', default='/home/jinze/vgg_distillation/checkpoint/ckpt_120_t.pth', type=str)
        parser.add_argument('--s_ckpt_path', default='/home/jinze/vgg_distillation/checkpoint/distill/ckpt_70_it.pth', type=str)
        parser.add_argument("--load_student", type=str2bool, default='True', help="use pre-train student model or not")
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--start_epoch', default=100, type=int)
        parser.add_argument("--weight-decay", type=float, default=1.e-4, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--it", type=str2bool, default='True', help="is inter model loss using or not")
        parser.add_argument("--ht", type=str2bool, default='False', help="is ht model loss using or not")
        parser.add_argument("--ce", type=str2bool, default='True', help="is ce loss using or not")
        parser.add_argument("--resume", type=str2bool, default='False', help="is remusing student ckpt or not")
        parser.add_argument("--lambda-ht", type=float, default=1.0, help="lambda_ht")
        parser.add_argument("--lambda-it", type=float, default=0.2, help="lambda_it")
        parser.add_argument("--lambda-ce", type=float, default=0.1, help="lambda_ce")
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        args = parser.parse_args()
        return args
