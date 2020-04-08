from networks.net import VGGNet
from utils.utils import *
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../../', train=True,
                                        download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=32,
                              shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../', train=False,
                                       download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=32,
                             shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
teacher = VGGNet(teacher_arch, 10)
print_model_parm_nums(teacher, 'teacher_model')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(teacher.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = teacher(inputs)
        loss = criterion(outputs[1], labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
