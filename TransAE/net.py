import torch
import torch.nn as nn


# linear block
class Liner_Module(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Liner_Module, self).__init__()
        self.liner = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        x = self.liner(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AE(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=8),
        )
        self.decoder = nn.Sequential(
            Liner_Module(input_dim=8, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            nn.Linear(128, output_dim),
        )

    def forward(self, input: torch.Tensor):
        x_feature = self.encoder(input)
        x = self.decoder(x_feature)
        return x, x_feature



