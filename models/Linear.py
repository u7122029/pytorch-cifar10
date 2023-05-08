import torch
import torch.nn as nn
import torch.nn.functional as func

class Linear(nn.Module):
    """
    The Linear Model
    """
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features=32*32*3, out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def linear(pretrained=True, device="cpu"):
    model = Linear()
    if pretrained:
        model = torch.hub.load_state_dict_from_url(
            "https://github.com/u7122029/pytorch-cifar10/releases/download/pretrained/linear.pth",
            map_location=device)
    return model

if __name__ == "__main__":
    print(linear())