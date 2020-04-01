import logging
import os


def load_s_model(args, model, with_module=True):
    logging.info('------------')
    if not os.path.exists(args.s_ckpt_path):
        os.makedirs(args.s_ckpt_path)
    if args.is_student_load_imgnet:
        pass


def load_t_model(args, ckpt_path):
    logging.info("------------")
    pass


def load_d_model(args, model, with_module=True):
    pass


def print_model_parm_nums(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    logging.info(string + ': Number of params: %.2fM', sum(b) / 1e6)