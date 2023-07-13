import timm
import torch.nn as nn



class EfficientFormer2(nn.Module):
    def __init__(self, config):
        super(EfficientFormer2, self).__init__()
        size = config.img_size
        self.model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=True)
        self.fc = nn.Linear(176 * size//32 * size//32, 2, bias=False)


    def forward(self, x):
        batch_size = x.size(0)
        output = self.model.forward_features(x)
        output = output.view(batch_size, -1)
        output = self.fc(output)

        return output
