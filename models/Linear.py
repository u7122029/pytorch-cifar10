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

if __name__ == "__main__":
    print(Linear())