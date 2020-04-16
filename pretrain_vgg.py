from networks.net import VGGNet
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch
import time

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../cifar10', train=True, 
                                        download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=32,
                              shuffle=True, num_workers=2)
PATH = './ckpt{}.pth'

print(len(trainloader))
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
"""
teacher_arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
512, 512, 512, 'M', 512, 512, 512, 'M']
"""
teacher = VGGNet(teacher_arch, 10)
teacher.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9)

for epoch in range(150):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, dat in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = dat
        inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = teacher(inputs)
        loss = criterion(outputs[1], labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        if epoch % 5 == 4:
            torch.save(teacher.state_dict(), PATH.format(int(time.time())))

print('Finished Training')
