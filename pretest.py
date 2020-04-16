from networks.net import VGGNet
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='../cifar10', train=False, download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=32,
                             shuffle=False, num_workers=2)

PATH = "./cifar_net.pth"
teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']

teacher_test = VGGNet(teacher_arch, 10)
teacher_test.load_state_dict(torch.load(PATH))
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for dat in testloader:
        images, labels = dat
        outputs = teacher_test(images)
        _, predicted = torch.max(outputs[1], 1)
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
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
print('Average Accuracy: %2d %%' % (accuracy / 10))