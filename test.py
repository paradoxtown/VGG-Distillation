from dataset.datasets import CSDataSet
import numpy as np
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils import data
import torch.nn.functional as F
from networks.net import VGGNet

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
# dataset = CSDataSet(root='../', list_path='dataset/list/cityscapes/train.lst', max_iter=40000 * 8,
#                     crop_size=(512, 512), mean=IMG_MEAN)
# image, label, size, name = dataset.__getitem__(0)
#
# image = torch.tensor([image])


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# conv1 = conv3x3(3, 64, stride=2)
# bn1 = nn.BatchNorm2d(64)
# relu1 = nn.ReLU(inplace=False)
# conv2 = conv3x3(64, 64)
# bn2 = nn.BatchNorm2d(64)
# relu2 = nn.ReLU(inplace=False)
# conv3 = conv3x3(64, 128)
# bn3 = nn.BatchNorm2d(128)
# relu3 = nn.ReLU(inplace=False)
# maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
# x = relu1(bn1(conv1(image)))
# x = relu2(bn2(conv2(x)))
# x = relu3(bn3(conv3(x)))
# x = maxpool(x)
# print(x.shape)
# print(x)

# CIFar10path = '../'
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = torchvision.datasets.CIFAR10(root=CIFar10path,
#                                              train=True,
#                                              transform=transform,
#                                              download=False)
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
# data_iter = iter(train_loader)
# images, labels = next(data_iter)
# index = 15
# image = images[index].numpy()
# print(image.shape)
# x = images
# print(x.shape)
# cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# cfgs = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
# cfgs = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
#
#
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
#
# def make_layers():
#     layers = []
#     in_channel = 3
#     for cfg in cfgs:
#         if cfg == 'M':
#             layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
#         else:
#             conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
#             layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channel = cfg
#     layers += [nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#                DoubleConv(32, 32),
#                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
#                DoubleConv(16, 8),
#                nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
#                DoubleConv(4, 3)]
#     return nn.Sequential(*layers)
#
#
# features = make_layers()
# classifier = nn.Sequential(
#     nn.Linear(in_features=64 * 4 * 4, out_features=4096),
#     nn.ReLU(inplace=True),
#     nn.Dropout(),
#     nn.Linear(256, 256),
#     nn.ReLU(inplace=True),
#     nn.Dropout(),
#     nn.Linear(256, 10),
#     nn.Softmax(dim=1)
# )
#
# feature = features(x)
# print(feature.shape)
# linear_input = torch.flatten(feature, 1)
# output = classifier(linear_input)
# print(output)
# print(output.shape)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='../', train=True,
                                        download=False, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=32,
                              shuffle=True, num_workers=2)

x, _ = next(iter(trainloader))
teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
teacher = VGGNet(teacher_arch, 10)
out = teacher(x)
