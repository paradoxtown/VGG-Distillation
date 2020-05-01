import torch.nn as nn

"""
teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
student_arch = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
"""


class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes

        cfgs1 = [32, 32, 32, 'M', 48, 48, 48, 48, 'M']
        in_channel = 3
        self.layers1 = []
        for cfg in cfgs1:
            if cfg == 'M':
                self.layers1 += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
                self.layers1 += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
                in_channel = cfg

        self.feature1 = nn.Sequential(*self.layers1)

        cfgs2 = [64, 64, 64, 64, 'M']
        self.layers2 = []
        for cfg in cfgs2:
            if cfg == 'M':
                self.layers2 += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
                self.layers2 += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
                in_channel = cfg

        self.feature2 = nn.Sequential(*self.layers2)

        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 10)
        )

    def forward(self, input_data):
        guided = self.feature1(input_data)
        feature = self.feature2(guided)
        guided = self.regressor(guided)
        feature_ = feature.view(-1, 1024)
        class_info = self.classifier(feature_)
        soft_result = nn.Softmax(dim=1)(class_info)
        return [guided, soft_result, class_info]


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes

        cfgs1 = [64, 64, 64, 'M', 96, 96, 96, 96, 'M']
        in_channel = 3
        self.layers1 = []
        for cfg in cfgs1:
            if cfg == 'M':
                self.layers1 += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
                self.layers1 += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
                in_channel = cfg

        self.feature1 = nn.Sequential(*self.layers1)

        cfgs2 = [128, 128, 128, 128, 'M']
        self.layers2 = []
        for cfg in cfgs2:
            if cfg == 'M':
                self.layers2 += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
                self.layers2 += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
                in_channel = cfg

        self.feature2 = nn.Sequential(*self.layers2)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 10)
        )

    def forward(self, input_data):
        hint = self.feature1(input_data)
        feature = self.feature2(hint)
        feature_ = feature.view(-1, 2048)
        class_info = self.classifier(feature_)
        soft_result = nn.Softmax(dim=1)(class_info)
        return [hint, soft_result, class_info]


if __name__ == '__main__':
    # decrease channel number
    student_cfgs = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
    teacher_cfgs = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
    student = SimpleNet(10)
    teacher = VGGNet(10)
