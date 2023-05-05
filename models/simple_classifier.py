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
            nn.Linear(in_features=in_size, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True)
        )
        self.way2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1000, bias=True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True)
        )
        
        self.cls = nn.Linear(in_features=1000, out_features=num_classes, bias=True)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        x = self.dropout(x)
        x = self.way2(x)
        x = self.dropout(x)
        logits = self.cls(x)
        
        return logits
    
    def intermediate_forward(self, x):
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.way1(x)
        x = self.dropout(x)
        x = self.way2(x)
        x = self.dropout(x)

        return x
