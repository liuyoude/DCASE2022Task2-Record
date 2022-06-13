import torch
import torch.nn as nn


class ASDLoss(nn.Module):
    def __init__(self):
        super(ASDLoss, self).__init__()
        self.rec_loss = nn.MSELoss()

    def forward(self, out, target):
        return self.rec_loss(out, target)