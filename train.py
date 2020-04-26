from networks.kd_model import NetModel
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import warnings
import time
from utils.config import Config

args = Config().initialize()
print(args)

warnings.filterwarnings("ignore")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = NetModel(args)
for epoch in range(args.epoches):
    total = 0
    correct = 0
    for i, dat in enumerate(trainloader, 0):
        model.set_input(dat)
        model.optimize_parameters()
        total, correct = model.evaluate_model(total, correct)
        if i % 100 == 99:
            model.print_info(epoch, i)
    if epoch % 10 == 9:
        model.save_ckpt(int(time.time()), epoch)
