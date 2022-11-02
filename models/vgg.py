import torch.nn as nn
import torch

class VGG(nn.Module): # VGG模型定义
    def __init__(self, num_classes=2, feature=16, init_weights=False):
        assert feature in [11, 13, 16, 19], 'VGG don\'t consist of {}'.format(feature)
        self.cfgs = { # 定义各种模型
            11: [64, 'M', 128, 'M', 256,256, 'M', 512, 512,'M', 512,512, 'M'],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        super(VGG, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 500),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(500, 20),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(20, num_classes)
        )
        self.backbone = self._make_feature(self.cfgs[feature]) 
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x): # 向前传播网络构建
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1) # 特征扁平化，也可以x.view(-1, 512*7*7)
        x = self.classifier(x)
        return x

    def _initialize_weights(self): # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def _make_feature(self, cfg: list): # 构建模型
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)
