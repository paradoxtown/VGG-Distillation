from networks.kd_model import NetModel
from torch.utils import data
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import warnings
import time
import random
from utils.config import Config

args = Config().initialize()
print(args)

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(7)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train)
# trainset = torchvision.datasets.CIFAR100(root='../cifar100', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

model = NetModel(args)
for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
    t1 = time.time()
    total = 0
    correct = 0
    model.adjust_learning_rate(model.optimizer, epoch)
    for i, dat in enumerate(trainloader, 0):
        model.set_input(dat)
        model.optimize_parameters()
        total, correct = model.evaluate_model(total, correct)
        if i % 500 == 499:
            model.print_info(epoch, i)
    t2 = time.time()
    print('--------> cost time: %.3f min' % ((t2 - t1) / 60.0))
    t1 = time.time()
    if epoch % 10 == 9:
        model.save_ckpt(int(time.time()), epoch)
