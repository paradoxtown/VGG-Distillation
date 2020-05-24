from networks.net import VGGNet, SimpleNet
from networks.net import VGGNet16, SimpleNet16A
from networks.net import VGGNet16s, SimpleNet16B, SimpleNet16C, SimpleNet16D
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch
import time
from utils.config import Config
from utils.utils import print_model_parm_nums

args = Config().initialize()
print(args)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train)
trainset = torchvision.datasets.CIFAR100(root='../cifar100', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

print(len(trainloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

is_teacher = False
is_16 = True
is_16s = True
if is_teacher:
    if is_16:
        if is_16s:
            PATH = './checkpoint/ckpt_vggs_{}_t.pth'
            net = VGGNet16s(100)
        else:
            PATH = './checkpoint/ckpt_vgg_{}_t.pth'
            net = VGGNet16(10)
    else:
        PATH = './checkpoint/ckpt_{}_t.pth'
        net = VGGNet(10)
    if args.resume:
        net.load_state_dict(torch.load(args.t_ckpt_path))
        print("load {} successfully".format(args.t_ckpt_path))
else:
    if is_16:
        if is_16s:
            PATH = './checkpoint/ckpt_vggs_{}_s.pth'
            net = SimpleNet16A(100)
            print('SimpleNet16C')
        else:
            PATH = './checkpoint/ckpt_vgg_{}_s.pth'
            net = SimpleNet16(10)
    else:
        PATH = './checkpoint/ckpt_{}_s.pth'
        net = SimpleNet(10)
    if args.resume:
        net.load_state_dict(torch.load(args.s_ckpt_path))
        print("load {} successfully".format(args.s_ckpt_path))
net.cuda()
print_model_parm_nums(net, "")

loss_ce = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.e-4)

def adjust_learning_rate(epoch):
        #     args = self.args
        #     lr = self.lr_poly(base_lr, i_iter, args.num_steps, args.power)
        #     optimizer.param_groups[0]['lr'] = lr
        #     return lr
        scale = 0.1
        lr_list = [0.1] * 100
        lr_list += [0.1 * scale] * 50
        lr_list += [0.1 * scale * scale] * 50

        lr = lr_list[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

for epoch in range(0, 200):  # loop over the dataset multiple times

    adjust_learning_rate(epoch)
    running_loss = 0.0
    running_l_ce = 0.0
    total = 0
    correct = 0
    time1 = time.time()
    for i, dat in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = dat
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = 0.1 * loss_ce(outputs[-2], labels)
        loss.backward()
        optimizer.step()

        # evaluation train
        _, predicted = outputs[-2].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:
            time2 = time.time()
            print('[%d, %5d] loss: %.3f, acc: %.3f%% {}'.format(time2-time1) %
                  (epoch + 1, i + 1, running_loss / 500, 100.*correct/total))
            time1 = time.time()
            running_loss = 0.0
    if epoch % 10 == 9:
        # torch.save(net.state_dict(), PATH.format(int(time.time()), epoch + 1))
        torch.save(net.state_dict(), PATH.format(epoch + 1))

print('Finished Training')
