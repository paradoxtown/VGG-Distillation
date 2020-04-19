import logging
import os
import torch
# import torchvision.models as models
# from networks.net import VGGNet


def to_tuple_str(str_first, gpu_num, str_ind):
    if gpu_num > 1:
        tmp = '('
        for cpu_ind in range(gpu_num):
            tmp += '(' + str_first + '[' + str(cpu_ind) + ']' + str_ind + ',)'
            if cpu_ind != gpu_num - 1:
                tmp += ', '
        tmp += ')'
    else:
        tmp = str_first + str_ind
    return tmp


def load_s_model(args, model, with_module=True):
    logging.info('------------')
    if not os.path.exists(args.s_ckpt_path):
        os.makedirs(args.s_ckpt_path)


def load_t_model(model, ckpt_path):
    logging.info("------------")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        logging.info('load' + str(ckpt_path))
    else:
        logging.info('=> no teacher checkpoint find')
    logging.info('------------')


def load_d_model(args, model, with_module=True):
    logging.info("------------")
    if args.D_resume:
        if not os.path.exists(args.D_ckpt_path):
            os.makedirs(args.D_ckpt_path)
        file = args.D_ckpt_path + '/model_best.pth.tar'
        if os.path.isfile(file):
            checkpoint = torch.load(file)
            args.start_epoch = checkpoint['epoch']
            args.best_mean_IU = checkpoint['best_mean_IU']
            state_dict = checkpoint['state_dict']
            if not with_module:
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
            else:
                new_state_dict = state_dict
            model.load_state_dict(new_state_dict)
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(
                file, checkpoint['epoch']))
        else:
            logging.info("=> checkpoint '{}' does not exit".format(file))
    logging.info("------------")


def print_model_parm_nums(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    logging.info(string + ': Number of params: %.2fM', sum(b) / 1e6)


def l2(feat):
    return (((feat ** 2).sum(dim=1)) ** 0.5).reshape(feat.shape[0], 1, feat.shape[1], feat.shape[2]) + 1e-8


def similarity(feature):
    feature = feature.float()
    tmp = l2(feature).detach()
    feature = feature / tmp
    feature = feature.reshape(feature.shape[0], feature.shape[1], -1)
    return torch.einsum('icm, icn -> imn', [feature, feature])


def sim_dis_compute(f_s, f_t):
    sim_err = ((similarity(f_t) - similarity(f_s)) ** 2) / ((f_t.shape[-1] * f_t.shape[-2]) ** 2) / f_t.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


# if __name__ == '__main__':
#     net_arch16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
#                   512, 'M', 512, 512, 512, 'M5', "FC1", "FC2", "FC"]
#     teacher = VGGNet(net_arch16, 10)
#     print(torch.nn.Sequential(*list(teacher.children())[:]))
#     model = models.vgg16(pretrained=False)
#     print(torch.nn.Sequential(*list(model.children())[:]))
