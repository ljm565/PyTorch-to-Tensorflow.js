import torch
import torch.nn as nn



class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return nn.ReLU6()(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return x * nn.ReLU6()(x + 3) / 6



def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]

        padding = (kernel_size - 1) // 2

        if use_se:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),
                
                # SE
                SELayer(inp, inp),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish(),
                
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish()
            )

    def forward(self, x):
        return self.conv(x)


class PPLCNet(nn.Module):
    def __init__(self):
        super(PPLCNet, self).__init__()
        self.cfgs = [
           # k,  c,  s, SE
            [3,  32, 1, 0],

            [3,  64, 2, 0],
            [3,  64, 1, 0],
            
            [3,  128, 2, 0],
            [3,  128, 1, 0],
            
            [5,  256, 2, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            
            [5,  512, 2, 1],
            [5,  512, 1, 1],
        ]
        self.scale = 0.25
        self.dropout_prob = 0.2

        input_channel = _make_divisible(16 * self.scale)
        layers = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), HardSwish()]

        block = DepSepConv
        for k, c, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * self.scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(input_channel, 1280, 1, 1, 0)
        self.hwish = HardSwish()
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.classifier = nn.Linear(1280, 2)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.hwish(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x