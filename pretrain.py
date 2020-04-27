from networks.net import SimpleNet
from networks.net import VGGNet
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch
import time

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
PATH = './checkpoint/ckpt_{}_{}.pth'

print(len(trainloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

is_teacher = False
if is_teacher:
    net = VGGNet(10)
else:
    net = SimpleNet(10)
    net.load_state_dict(torch.load('/home/jinze/vgg_distillation/checkpoint/ckpt_1587984581_90.pth'))
net.cuda()

loss_ce = nn.CrossEntropyLoss()
loss_up = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1.e-4)

for epoch in range(90, 200):  # loop over the dataset multiple times

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
        loss = 0.1 * loss_ce(outputs[2], labels)
        loss.backward()
        optimizer.step()

        # evaluation train
        _, predicted = outputs[2].max(1)
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
        torch.save(net.state_dict(), PATH.format(int(time.time()), epoch + 1))

print('Finished Training')
