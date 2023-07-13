import torch.nn as nn
from torchvision.models import resnet18



class ResNet(nn.Module):
    def __init__(self, config):
        super(ResNet, self).__init__()
        size = config.img_size//8 if config.img_size % 8 == 0 else config.img_size//8 + 1
        base_model = resnet18(pretrained=True, progress=False)
        base_model = list(base_model.children())[:-4]
        self.model = nn.Sequential(*base_model)
        self.fc = nn.Linear(128 * size * size, 2)


    def forward(self, x):
        batch_size = x.size(0)
        output = self.model(x)
        output = output.view(batch_size, -1)
        output = self.fc(output)
        return output
    


class ResNetTiny(nn.Module):
    def __init__(self, config):
        super(ResNetTiny, self).__init__()
        size = config.img_size//4 if config.img_size % 4 == 0 else config.img_size//4 + 1
        base_model = resnet18(pretrained=True, progress=False)
        base_model = list(base_model.children())[:-5]
        self.model = nn.Sequential(*base_model)
        self.fc = nn.Linear(64 * size * size, 2)


    def forward(self, x):
        batch_size = x.size(0)
        output = self.model(x)
        output = output.view(batch_size, -1)
        output = self.fc(output)
        return output