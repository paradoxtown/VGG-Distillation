from networks.net import SimpleNet
from networks.net import VGGNet
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch
# import os

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='../cifar10', train=False, download=True, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# files = os.listdir('./checkpoint')
PATH = '/home/jinze/vgg_distillation/checkpoint/distill/ckpt_200_it.pth'
print(PATH)

is_teacher = False
if is_teacher:
    net = VGGNet(10)
else:
    net = SimpleNet(10)
net.cuda()
net.load_state_dict(torch.load(PATH))
net.eval()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for dat in testloader:
        images, labels = dat
        images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs[2], 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

accuracy = 0
total = 0
for i in range(10):
    accuracy += 100 * class_correct[i] / class_total[i]
    print('Accuracy of %5s : %.4f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
print('Average Accuracy: %.4f %%' % (accuracy / 10))
