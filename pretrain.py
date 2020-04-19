from networks.net import SimpleNet
# from networks.net import VGGNet
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
teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
student_arch = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']

# net = VGGNet(teacher_arch, 10)
net = SimpleNet(student_arch, 10)
net.cuda()

loss_ce = nn.CrossEntropyLoss()
loss_up = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(150):  # loop over the dataset multiple times

    running_loss = 0.0
    running_l_ce = 0.0
    running_l_up = 0.0
    total = 0
    correct = 0
    for i, dat in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = dat
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        l_ce = loss_ce(outputs[1], labels)
        l_up = loss_up(outputs[2], inputs)
        loss = 0.1 * l_ce + l_up
        loss.backward()
        optimizer.step()

        # evaluation train
        _, predicted = outputs[1].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # print statistics
        running_loss += loss.item()
        running_l_ce += l_ce.item()
        running_l_up += l_up.item()
        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f, loss_ce: %.3f, loss_up: %.3f, acc: %.3f%%' %
                  (epoch + 1, i + 1, running_loss / 500, running_l_ce / 500, running_l_up / 500, 100.*correct/total))
            running_loss = 0.0
            running_l_ce = 0.0
            running_l_up = 0.0
    if epoch % 5 == 4:
        torch.save(net.state_dict(), PATH.format(int(time.time()), epoch))

print('Finished Training')
