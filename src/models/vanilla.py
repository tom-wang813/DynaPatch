import torch.nn as nn

class VanillaModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, x):
        return self.model(x)
