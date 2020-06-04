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
        parser.add_argument('--t_ckpt_path', default='/content/drive/My Drive/vgg16_distillation/checkpoint/pretrain/ckpt_vggs_200_t_32.pth', type=str)
        parser.add_argument('--s_ckpt_path', default='/content/drive/My Drive/vgg16_distillation/checkpoint/distill/ht/FT-2-A/ckpt_ht.pth', type=str)
        parser.add_argument("--load_student", type=str2bool, default='False', help="use pre-train student model or not")
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--start_epoch', default=0, type=int)
        parser.add_argument("--weight-decay", type=float, default=1.e-4, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--it", type=str2bool, default='False', help="")
        parser.add_argument("--st", type=str2bool, default='False', help="mode: soft-target")
        parser.add_argument("--lg", type=str2bool, default='False', help="mode: logits")
        parser.add_argument("--sp", type=str2bool, default='False', help="mode: similarity preserving")
        parser.add_argument("--at", type=str2bool, default='True', help="mode: attention transfer")
        parser.add_argument("--ht", type=str2bool, default='False', help="mode: fitnets")
        parser.add_argument("--ce", type=str2bool, default='True', help="mode: task")
        parser.add_argument("--resume", type=str2bool, default='False', help="whether resuming student ckpt")
        parser.add_argument("--lambda-ht", type=float, default=1.0, help="lambda_ht")
        parser.add_argument("--lambda-it", type=float, default=0.1, help="lambda_it")
        parser.add_argument("--lambda-st", type=float, default=0.1, help="lambda_st")
        parser.add_argument("--lambda-lg", type=float, default=0.1, help="lambda_lg")
        parser.add_argument("--lambda-sp", type=float, default=30.0, help="lambda_sp")
        parser.add_argument("--lambda-at", type=float, default=10.0, help="lambda_at")
        parser.add_argument("--lambda-ce", type=float, default=0.1, help="lambda_ce")
        parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
        args = parser.parse_args()
        return args
