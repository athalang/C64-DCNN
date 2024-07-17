import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified LeNeT-5 architecture
# FC parameters divided by 8 as suggested in LightweightNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2, bias=False)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(400, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    try:
        from torchinfo import summary

        model = Net()
        batch_size = 64
        summary(model, input_size=(batch_size, 1, 28, 28))
    except ImportError:
        print("torchinfo not found; skipping summary")