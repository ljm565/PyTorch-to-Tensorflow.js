import torch.nn as nn



class LN(nn.Module):
    def __init__(self, config):
        super(LN, self).__init__()
        self.fc = nn.Linear(config.img_size * config.img_size * 3, 2)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
