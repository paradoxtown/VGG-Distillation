import torch.nn as nn


class SimpleNet(nn.Module):
    """
    arch = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
    """
    def __init__(self, cfgs, num_classes):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes
        s_layers = []
        in_channel = 3

        # down sampling
        for cfg in cfgs:
            if cfg == 'M':
                s_layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
                s_layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = cfg

        self.feature = nn.Sequential(*s_layers)

        # up sampling
        up_layers = [nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                     DoubleConv(32, 32),
                     nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                     DoubleConv(16, 8),
                     nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
                     DoubleConv(4, 3)]

        self.restore = nn.Sequential(*up_layers)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, input_data):
        feature = self.feature(input_data)
        soft_result = self.classifier(feature)
        restore = self.restore(feature)
        return [feature, soft_result, restore]


class VGGNet(nn.Module):
    def __init__(self, cfgs, num_classes):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        vgg_layers = []
        in_channel = 3

        # down sampling
        for cfg in cfgs:
            if cfg == 'M':
                vgg_layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
                vgg_layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = cfg

        self.feature = nn.Sequential(*vgg_layers)

        # up sampling
        up_layers = [nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                     DoubleConv(64, 32),
                     nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                     DoubleConv(16, 8),
                     nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2),
                     DoubleConv(4, 3)]

        self.restore = nn.Sequential(*up_layers)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, input_data):
        """
        arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
        """
        feature = self.feature(input_data)
        feature_ = feature.view(-1, 2048)
        print(feature_.shape)
        soft_result = self.classifier(feature_)
        restore = self.restore(feature)
        return [feature, soft_result, restore]


class DoubleConv(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


if __name__ == '__main__':
    # decrease channel number
    student_cfgs = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
    teacher_cfgs = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
    student = SimpleNet(student_cfgs, 10)
    teacher = VGGNet(teacher_cfgs, 10)
