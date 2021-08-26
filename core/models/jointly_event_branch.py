import torch
import torch.nn as nn
from core.models.common_layers import Stage
from core.config import cfg

class Reshape0(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 1, 3, 2)
        

class Reshape1(torch.nn.Module):
    def forward(self, x):
        x2 = x.permute(3, 0, 1, 2)
        x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
        return x2

class Reshape2(torch.nn.Module):
    def forward(self, x):
        return x[0].permute(1, 0, 2)
        

class EventBranch(nn.Module):
    def __init__(self, in_dim, out_dim, weights='DeepLab', *args, **kwargs):
        super(EventBranch, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc_id = "head.10"
        self.base = lambda x: x  # required by NAS
        
        self.stages = []
        layers = []
        stages = [
            (256, [
                Reshape0(),
                nn.MaxPool2d((8, 1)),
                nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
                nn.ReLU(inplace=True)
            ]),
            (256, [
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
                nn.ReLU(inplace=True)
            ]),
            (128, [
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(128, eps=1e-03, momentum=0.05),
                nn.ReLU(inplace=True)
            ]),
        ]

        for channels, stage in stages:
            layers += stage
            self.stages.append(Stage(channels, stage))
        self.stages = nn.ModuleList(self.stages)

        # Used for backward compatibility with weight loading
        self.features = nn.Sequential(*layers)

        head = [
            Reshape1(),
            nn.GRU(256, 64, 1, bidirectional=True),
            Reshape2(),
            nn.Linear(64 * 2, 32),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(32, 25)
        ]
        self.head = nn.Sequential(*head)

        self.weights = weights
        self.init_weights()

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

    def init_weights(self):
        for layer in self.head.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)