# import torch
# import torch.nn as nn
# import torch.optim as optim

# from common.utils import *
# from common.train_utils import *


# class Net(nn.Module):
#     def __init__(self, num_classes: int = 10) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, 3, 1, padding="same")
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, 3, 1, padding="same")
#         self.bn4 = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(1024, 256)
#         self.drop = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = torch.relu(F.max_pool2d(self.conv1(x), 2))  # 16 x 16 x 32
#         x = self.bn1(x)
#         x = torch.relu(F.max_pool2d(self.conv2(x), 2))  # 8 x 8 x 64
#         x = self.bn2(x)
#         x = torch.relu(F.max_pool2d(self.conv3(x), 2))  # 4 x 4 x 128
#         x = self.bn3(x)
#         x = torch.relu(F.max_pool2d(self.conv4(x), 2))  # 2 x 2 x 256
#         x = self.bn4(x)
#         x = x.view(x.size(0), -1)
#         x = self.drop(x)
#         x = torch.relu(self.fc1(x))
#         x = torch.log_softmax(self.fc2(x), dim=1)
#         return x




# def main() -> None:
#     # Load the data
#     train_loader, test_loader = get_data('cifar10', batch_size=256)

#     # Create a model
#     model = Net()
#     print("Model Parameter Count:", sum(p.numel() for p in model.parameters()))

#     # Create an optimizer
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Train the model
#     train(model, train_loader, optimizer, epochs=75)

#     # will save the model
#     # torch.save(model.state_dict(), "./weights/v9-cifar10.pth")

#     # Evaluate the model
#     test_loss, test_acc = evaluate(model, test_loader)
#     print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import *
from common.train_utils import *


class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, padding="same")
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))  # 16 x 16 x 32
        x = self.bn1(x)
        x = torch.relu(F.max_pool2d(self.conv2(x), 2))  # 8 x 8 x 64
        x = self.bn2(x)
        x = torch.relu(F.max_pool2d(self.conv3(x), 2))  # 4 x 4 x 128
        x = self.bn3(x)
        x = torch.relu(F.max_pool2d(self.conv4(x), 2))  # 2 x 2 x 256
        x = self.bn4(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x
    
class ResBlockA(nn.Module):

    def __init__(self, in_chann, chann, stride):
        super(ResBlockA, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.functional.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        
        if (x.shape == y.shape):
            z = x
        else:
            z = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)            

            x_channel = x.size(1)
            y_channel = y.size(1)
            ch_res = (y_channel - x_channel)//2

            pad = (0, 0, 0, 0, ch_res, ch_res)
            z = nn.functional.pad(z, pad=pad, mode="constant", value=0)

        z = z + y
        z = nn.functional.relu(z)
        return z


class PlainBlock(nn.Module):

    def __init__(self, in_chann, chann, stride):
        super(PlainBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.functional.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        y = nn.functional.relu(y)
        return y


class BaseNet(nn.Module):
    
    def __init__(self, Block, n):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn0   = nn.BatchNorm2d(16)
        self.convs  = self._make_layers(n)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = nn.functional.relu(x)
        
        x = self.convs(x)
        
        x = self.avgpool(x)

        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

    def _make_layers(self, n):
        layers = []
        in_chann = 16
        chann = 16
        stride = 1
        for i in range(3):
            for j in range(n):
                if ((i > 0) and (j == 0)):
                    in_chann = chann
                    chann = chann * 2
                    stride = 2

                layers += [self.Block(in_chann, chann, stride)]

                stride = 1
                in_chann = chann

        return nn.Sequential(*layers)


def ResNet(n):
    return BaseNet(ResBlockA, n)

def PlainNet(n):
    return BaseNet(PlainBlock, n)

# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

transform_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
        ])

def main() -> None:
    # Load the data
    train_loader, test_loader = get_data('cifar10', batch_size=32, transform=transform_func)

    # Create a model
    model = ResNet(18)
    print("Model Parameters Count :", sum(p.numel() for p in model.parameters()))

    # Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, optimizer, epochs=50)

    # will save the model
    # torch.save(model.state_dict(), "./weights/v9-cifar10.pth")
    torch.save(model.state_dict(), "./weights/v9-cifar10-resnet-110.pth")

    # Evaluate the model
    test_loss, test_acc = evaluate(model, test_loader)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
