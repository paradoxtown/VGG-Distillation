import torch.nn as nn

"""
teacher_arch = [64, 64, 64, 'M', 96, 96, 96, 96, 'M', 128, 128, 128, 128, 'M']
student_arch = [32, 32, 32, 'M', 48, 48, 48, 48, 'M', 64, 64, 64, 64, 'M']
vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
simple_vgg16 = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
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
        guided_ = self.feature1(input_data)
        feature = self.feature2(guided_)
        guided = self.regressor(guided_)
        feature_ = feature.view(-1, 1024)
        class_info = self.classifier(feature_)
        soft_result = nn.Softmax(dim=1)(class_info)
        return [guided, soft_result, class_info, guided_]


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
        return [hint, soft_result, class_info, hint]


class SimpleNet16(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet16, self).__init__()
        self.num_classes = num_classes

        cfgs1 = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M']
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

        cfgs2 = [256, 256, 256, 'M', 256, 256, 256, 'M']
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
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, input_data):
        guided_ = self.feature1(input_data)
        feature = self.feature2(guided_)
        guided = self.regressor(guided_)
        feature_ = feature.view(feature.size(0), -1)
        class_info = self.classifier(feature_)
        soft_result = nn.Softmax(dim=1)(class_info)
        return [guided, soft_result, class_info, guided_, feature]


class VGGNet16(nn.Module):
    # 0.1 100, 0.01 40, 0.001
    def __init__(self, num_classes):
        super(VGGNet16, self).__init__()
        self.num_classes = num_classes

        cfgs1 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
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

        cfgs2 = [512, 512, 512, 'M', 512, 512, 512, 'M']
        self.layers2 = []
        for cfg in cfgs2:
            if cfg == 'M':
                self.layers2 += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=in_channel, out_channels=cfg, kernel_size=3, padding=1)
                self.layers2 += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
                in_channel = cfg
        self.layers2.append(nn.AvgPool2d(kernel_size=1, stride=1))
        self.feature2 = nn.Sequential(*self.layers2)

        self.classifier = nn.Linear(512, 10)

    def forward(self, input_data):
        hint = self.feature1(input_data)
        feature = self.feature2(hint)
        feature_ = feature.view(feature.size(0), -1)
        class_info = self.classifier(feature_)
        soft_result = nn.Softmax(dim=1)(class_info)
        # return [hint, feature, soft_result]
        return [hint, soft_result, class_info, hint, feature]


class SimpleNet16s(nn.Module):
    def __init__(self, num_classes):
        super(SimpleNet16s, self).__init__()
        self.num_classes = num_classes
        self.in_channel = 3

        self.m1 = nn.Sequential(*self.make_layer([16, 16, 'M']))
        self.m2 = nn.Sequential(*self.make_layer([32, 32, 'M']))
        self.m3 = nn.Sequential(*self.make_layer([64, 64, 64, 'M']))
        self.m4 = nn.Sequential(*self.make_layer([128, 128, 128, 'M']))
        self.m5 = nn.Sequential(*self.make_layer([128, 128, 128, 'M']))

        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(128, 10)

    def make_layer(self, cfgs):
        layers = []
        for cfg in cfgs:
            if cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=self.in_channel, out_channels=cfg, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
                self.in_channel = cfg
        return layers

    def forward(self, input_data):
        o1 = self.m1(input_data)
        o2 = self.m2(o1)
        o3 = self.m3(o2)
        guided = self.regressor(o3)
        o4 = self.m4(o3)
        o5 = self.m5(o4)
        feature = o5.view(o5.size(0), -1)
        class_info = self.classifier(feature)
        soft_target = nn.Softmax(dim=1)(class_info)
        return [o1, o2, o3, o4, o5, guided, feature, class_info, soft_target]


class VGGNet16s(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet16s, self).__init__()
        self.num_classes = num_classes
        self.in_channel = 3

        self.m1 = nn.Sequential(*self.make_layer([64, 64, 'M']))
        self.m2 = nn.Sequential(*self.make_layer([128, 128, 'M']))
        self.m3 = nn.Sequential(*self.make_layer([256, 256, 256, 'M']))
        self.m4 = nn.Sequential(*self.make_layer([512, 512, 512, 'M']))
        self.m5 = nn.Sequential(*self.make_layer([512, 512, 512, 'M']))

        self.classifier = nn.Linear(512, 10)

    def make_layer(self, cfgs):
        layers = []
        for cfg in cfgs:
            if cfg == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            else:
                conv2d = nn.Conv2d(in_channels=self.in_channel, out_channels=cfg, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(cfg), nn.ReLU(inplace=True)]
                self.in_channel = cfg
        return layers

    def forward(self, input_data):
        o1 = self.m1(input_data)
        o2 = self.m2(o1)
        o3 = self.m3(o2)
        hint = o3
        o4 = self.m4(o3)
        o5 = self.m5(o4)
        feature = o5.view(o5.size(0), -1)
        class_info = self.classifier(feature)
        soft_target = nn.Softmax(dim=1)(class_info)
        return [o1, o2, o3, o4, o5, hint, feature, class_info, soft_target]
