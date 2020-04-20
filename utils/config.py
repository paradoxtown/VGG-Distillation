import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config:
    def initialize(self):
        parser = argparse.ArgumentParser(description='knowledge-distillation')
        parser.add_argument('--num_classes', default=10, type=int)
        parser.add_argument('--t_ckpt_path', default='/home/jinze/vgg_distillation/checkpoint/ckpt_1587386983_100.pth', type=str)
        parser.add_argument('--s_ckpt_path', default='/home/jinze/vgg_distillation/checkpoint/distill/', type=str)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--epoches', default=20, type=int)
        parser.add_argument("--weight-decay", type=float, default=1.e-4, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--pi", type=str2bool, default='False', help="is pixel wise loss using or not")
        parser.add_argument("--it", type=str2bool, default='True', help="is iter model loss using or not")
        parser.add_argument("--lambda-pi", type=float, default=1.0, help="lambda_pi")
        parser.add_argument("--lambda-it", type=float, default=1.0, help="lambda_it")
        parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
        args = parser.parse_args()
        return args
