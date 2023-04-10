import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

# Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, in_size=2880, num_classes=20) -> None:
        super(SimpleClassifier, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.way1 = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        
        self.cls = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        logits = self.cls(x)
        
        return logits
    
    def intermediate_forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        
        return x
        
class SimpleSegClssifier(nn.Module):
    def __init__(self, in_size=2880, num_classes=21):

        super(SimpleSegClssifier, self).__init__()
        self.depthway1 = nn.Sequential(
            nn.Conv2d(in_size, 1000, kernel_size=1),
            nn.GroupNorm(4,1000),
            nn.ReLU(inplace=True),
        )
        self.depthway2 = nn.Sequential(
            nn.Conv2d(1000, 1000, kernel_size=1),
            nn.GroupNorm(4,1000),
            nn.ReLU(inplace=True),
        )
        self.depthway3 = nn.Sequential(
            nn.Conv2d(1000, 512, kernel_size=1),
            nn.GroupNorm(4,512),
            nn.ReLU(inplace=True),
        )

        self.clsdepth = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        seg = self.depthway1(x)
        seg = self.depthway2(seg)
        seg = self.depthway3(seg)
        seg = self.clsdepth(seg)

        return seg