import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_features = self.layers(x)
        y_features = self.layers(y)
        return F.mse_loss(x_features, y_features)