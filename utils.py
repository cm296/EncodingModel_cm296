
from torch import nn
import torch
from torchvision.models.alexnet import alexnet

## Specify Alexnet model
class AlexNet(nn.Module):
    def __init__(self, feature_name):
        super().__init__()
        self.feature_name = feature_name
        base = alexnet(pretrained=True)
        self.conv_1 = base.features[:3]
        self.conv_2 = base.features[3:6]
        self.conv_3 = base.features[6:8]
        self.conv_4 = base.features[8:10]
        self.conv_5 = base.features[10:]
        self.avgpool = base.avgpool
        self.fc_1 = base.classifier[:3]
        self.fc_2 = base.classifier[3:6]
        self.fc_3 = base.classifier[6:]
        self.eval()
    def forward(self, stimuli):
        x = self.conv_1(stimuli)
        if 'conv_1' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_2(x)
        if 'conv_2' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_3(x)
        if 'conv_3' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_4(x)
        if 'conv_4' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_5(x)
        if 'conv_5' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        if 'pool' == self.feature_name: return x
        x = self.fc_1(x)
        if 'fc_1' == self.feature_name: return x
        x = self.fc_2(x)
        if 'fc_2' == self.feature_name: return x
        x = self.fc_3(x)
        if 'fc_3' == self.feature_name: return x
        return None 