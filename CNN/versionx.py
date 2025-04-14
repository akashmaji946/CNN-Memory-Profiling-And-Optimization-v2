
import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import *
from common.train_utils import *

class BaseNet(nn.Module):
    
    def __init__(self, Block, n, num_classes=1000):  # 1000 for ImageNet
        super(BaseNet, self).__init__()
        self.Block = Block
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # use standard conv
        self.bn0   = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.convs = self._make_layers(n)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # adapt to any input size
        self.fc = nn.Linear(64 * 4, num_classes)     # output channels of last block

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.convs(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def _make_layers(self, n):
        layers = []
        in_chann = 64
        chann = 64
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

    
class ResBlockA(nn.Module):
    def __init__(self, in_chann, chann, stride):
        super(ResBlockA, self).__init__()
        
        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(chann)

        # projection shortcut if needed (1x1 conv)
        self.downsample = None
        if stride != 1 or in_chann != chann:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chann, chann, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(chann)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)

        return out



def ResNet(n, num_classes=1000):
    return BaseNet(ResBlockA, n, num_classes=num_classes)


import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_imagenet(data_dir, batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader

def evaluate(model, data_loader):
    model.eval()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Testing"):
            
            data, target = batch
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)

            total_loss += loss.item() * data.size(0)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)

    return total_loss / total_samples, total_correct / total_samples

from torchvision import models
def main():
    data_path = "/home/akash/Desktop/ML-2/imagenet-mini"
    train, test_loader = get_data_imagenet(data_path, batch_size=128)

    # Load a pre-trained model (e.g., ResNet-18 from torchvision)
    model = models.resnet18(pretrained=True)  # You can replace this with ResNet-20 if you have it
    model.to(get_device())  # Ensure the model is loaded on the correct device
    print("Model Parameters Count:", sum(p.numel() for p in model.parameters()))

    # Evaluate the model on the test dataset
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
    

