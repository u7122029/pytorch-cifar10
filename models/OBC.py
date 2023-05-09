import torch
import torch.nn as nn
import torch.nn.functional as func

class OBC(nn.Module):
    """
    The One BrainCell Model
    Takes the mean pixel value of an image for classification.
    """
    def __init__(self):
        super(OBC, self).__init__()
        self.fc = nn.Linear(in_features=1, out_features=10)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = torch.mean(x,[1,2,3]).reshape(-1,1).to(self.device)
        x = self.fc(x)
        return x

def obc(pretrained=True, device="cpu"):
    model = OBC()
    if pretrained:
        model = torch.hub.load_state_dict_from_url(
            "https://github.com/u7122029/pytorch-cifar10/releases/download/pretrained/obc.pth",
            map_location=device)
    return model

if __name__ == "__main__":
    print(obc(True))