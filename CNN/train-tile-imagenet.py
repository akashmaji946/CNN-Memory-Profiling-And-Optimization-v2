# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from common.utils import *
# from common.train_utils import *
# import torch
# import torch.nn.functional as F

# import torch
# import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint

# def extract_tile_with_overlap(x, top, left, tile_size, padding):
#     B, C, H, W = x.shape
#     top_pad = max(padding - top, 0)
#     left_pad = max(padding - left, 0)
#     bottom_pad = max(padding - (H - (top + tile_size)), 0)
#     right_pad = max(padding - (W - (left + tile_size)), 0)

#     top_idx = max(top - padding, 0)
#     left_idx = max(left - padding, 0)
#     bottom_idx = min(top + tile_size + padding, H)
#     right_idx = min(left + tile_size + padding, W)

#     tile = x[:, :, top_idx:bottom_idx, left_idx:right_idx]
#     tile = F.pad(tile, (left_pad, right_pad, top_pad, bottom_pad))
#     return tile

# def tiled_conv2d(x, conv_layer, tile_size, use_checkpoint=False):
#     B, C, H, W = x.shape
#     padding = conv_layer.padding[0]  # assumes square padding
#     stride = conv_layer.stride[0]
#     kernel_size = conv_layer.kernel_size[0]
#     out_channels = conv_layer.out_channels

#     output = torch.zeros(B, out_channels, H, W, device=x.device, dtype=x.dtype)

#     for top in range(0, H, tile_size):
#         for left in range(0, W, tile_size):
#             tile = extract_tile_with_overlap(x, top, left, tile_size, padding)

#             # Use checkpointing to save memory
#             if use_checkpoint and x.requires_grad:
#                 tile_out = checkpoint(conv_layer, tile)
#             else:
#                 tile_out = conv_layer(tile)

#             h_out = min(tile_size, H - top)
#             w_out = min(tile_size, W - left)

#             top_pad = max(padding - top, 0)
#             left_pad = max(padding - left, 0)

#             h_out_conv = (h_out + 2 * padding - kernel_size) // stride + 1
#             w_out_conv = (w_out + 2 * padding - kernel_size) // stride + 1

#             output_slice = output[:, :, top:top + h_out_conv, left:left + w_out_conv]
#             tile_out_slice = tile_out[:, :, top_pad:top_pad + h_out_conv, left_pad:left_pad + w_out_conv]

#             assert output_slice.shape == tile_out_slice.shape, f"Shape mismatch: {output_slice.shape} vs {tile_out_slice.shape}"
#             output_slice.copy_(tile_out_slice)

#     return output

# class ResBlockA(nn.Module):
#     def __init__(self, in_chann, chann, stride, tile_size=None, use_checkpoint=False):
#         super(ResBlockA, self).__init__()
#         self.tile_size = tile_size
#         self.use_checkpoint = use_checkpoint

#         self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(chann)

#         self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(chann)

#         self.downsample = None
#         if stride != 1 or in_chann != chann:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_chann, chann, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(chann)
#             )

#     def forward(self, x):
#         identity = x

#         if self.tile_size:
#             out = tiled_conv2d(x, self.conv1, self.tile_size, use_checkpoint=self.use_checkpoint)
#         else:
#             out = self.conv1(x)

#         out = self.bn1(out)
#         out = torch.relu(out)

#         if self.tile_size:
#             out = tiled_conv2d(out, self.conv2, self.tile_size, use_checkpoint=self.use_checkpoint)
#         else:
#             out = self.conv2(out)

#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = torch.relu(out)
#         return out



# class BaseNet(nn.Module):
#     def __init__(self, Block, n, num_classes=1000, tile_size=None):
#         super(BaseNet, self).__init__()
#         self.Block = Block
#         self.tile_size = tile_size

#         self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # no tiling
#         self.bn0 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.convs = self._make_layers(n)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(64 * 4, num_classes)

#     def forward(self, x):
#         x = self.conv0(x)
#         x = self.bn0(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.convs(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

#     def _make_layers(self, n):
#         layers = []
#         in_chann = 64
#         chann = 64
#         stride = 1
#         for i in range(3):
#             for j in range(n):
#                 if i > 0 and j == 0:
#                     in_chann = chann
#                     chann = chann * 2
#                     stride = 2
#                 else:
#                     stride = 1

#                 layers.append(
#                     self.Block(in_chann, chann, stride, tile_size=self.tile_size, use_checkpoint=True)
#                 )
#                 in_chann = chann
#         return nn.Sequential(*layers)



# def ResNet(n, num_classes=1000, tile_size=None):
#     return BaseNet(ResBlockA, n, num_classes=num_classes, tile_size=tile_size)



# import argparse
# import torch
# import torch.nn.functional as F

# from common.utils import get_device, get_data
# # from main import ResNet, transform_func  # make sure this is correct for your structure
# import re
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
#     return parser.parse_args()

# def extract_depth_from_filename(weights_path):
#     match = re.search(r"resnet-tile-(\d+)", weights_path)
#     if match:
#         depth = int(match.group(1))
#         if (depth - 2) % 6 != 0:
#             raise ValueError(f"Invalid ResNet depth in file: {weights_path}")
#         return (depth - 2) // 6  # Compute n
#     else:
#         raise ValueError(f"Could not parse depth from file name: {weights_path}")


# import pynvml
# import torch.cuda.nvtx as nvtx
# import time

# # Function to log GPU stats
# def log_gpu_stats(label=""):
#     allocated = torch.cuda.memory_allocated() / (1024 ** 2)
#     reserved = torch.cuda.memory_reserved() / (1024 ** 2)
#     return allocated, reserved

# # Function to setup NVML
# def setup_nvml():
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     return handle

# # Function to log NVML stats
# def log_nvml(handle, label=""):
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     util = pynvml.nvmlDeviceGetUtilizationRates(handle)
#     return util.gpu, mem_info.used / (1024 ** 2)

# def get_data_imagenet(data_dir, batch_size):
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])
#     val_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ])

#     train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
#     val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

#     return train_loader, val_loader



# def main() -> None:
#     # Load the data
#     train_loader, test_loader = get_data_imagenet("/home/akash/Desktop/ML-2/imagenet-mini", batch_size=128)

#     # Create a model
#     model = ResNet(n=3, num_classes=1000, tile_size=16)  # Specify the tile size here
#     print("Model Parameters Count :", sum(p.numel() for p in model.parameters()))

#     # Create an optimizer
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Train the model
#     train(model, train_loader, optimizer, epochs=1)

#     # will save the model
#     # torch.save(model.state_dict(), "./weights/v9-cifar10.pth")
#     torch.save(model.state_dict(), "./weights-imagenet/imagenet-resnet-tile-20.pth")

#     # Evaluate the model
#     test_loss, test_acc = evaluate(model, test_loader)

#     print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
import os
from common.utils import get_device
from typing import Tuple
from tqdm import tqdm
import re
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import pynvml
import torch.cuda.nvtx as nvtx

def extract_tile_with_overlap(x, top, left, tile_size, padding):
    B, C, H, W = x.shape
    top_pad = max(padding - top, 0)
    left_pad = max(padding - left, 0)
    bottom_pad = max(padding - (H - (top + tile_size)), 0)
    right_pad = max(padding - (W - (left + tile_size)), 0)

    top_idx = max(top - padding, 0)
    left_idx = max(left - padding, 0)
    bottom_idx = min(top + tile_size + padding, H)
    right_idx = min(left + tile_size + padding, W)

    tile = x[:, :, top_idx:bottom_idx, left_idx:right_idx]
    tile = F.pad(tile, (left_pad, right_pad, top_pad, bottom_pad))
    return tile


def tiled_conv2d(x, conv_layer, tile_size, use_checkpoint=False):
    B, C, H, W = x.shape
    padding = conv_layer.padding[0]  # assumes square padding
    stride = conv_layer.stride[0]
    kernel_size = conv_layer.kernel_size[0]
    out_channels = conv_layer.out_channels

    output = torch.zeros(B, out_channels, H, W, device=x.device, dtype=x.dtype)

    for top in range(0, H, tile_size):
        for left in range(0, W, tile_size):
            tile = extract_tile_with_overlap(x, top, left, tile_size, padding)

            if use_checkpoint and x.requires_grad:
                tile_out = checkpoint(conv_layer, tile)
            else:
                tile_out = conv_layer(tile)

            h_out = min(tile_size, H - top)
            w_out = min(tile_size, W - left)

            top_pad = max(padding - top, 0)
            left_pad = max(padding - left, 0)

            h_out_conv = (h_out + 2 * padding - kernel_size) // stride + 1
            w_out_conv = (w_out + 2 * padding - kernel_size) // stride + 1

            output[:, :, top:top + h_out_conv, left:left + w_out_conv] = tile_out[:, :, top_pad:top_pad + h_out_conv, left_pad:left_pad + w_out_conv]

    return output


class ResBlockA(nn.Module):
    def __init__(self, in_chann, chann, stride, tile_size=None, use_checkpoint=False):
        super(ResBlockA, self).__init__()
        self.tile_size = tile_size
        self.use_checkpoint = use_checkpoint

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(chann)

        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(chann)

        self.downsample = None
        if stride != 1 or in_chann != chann:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chann, chann, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(chann)
            )

    def forward(self, x):
        identity = x

        if self.tile_size:
            out = tiled_conv2d(x, self.conv1, self.tile_size, use_checkpoint=self.use_checkpoint)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = torch.relu(out)

        if self.tile_size:
            out = tiled_conv2d(out, self.conv2, self.tile_size, use_checkpoint=self.use_checkpoint)
        else:
            out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = torch.relu(out)
        return out


class BaseNet(nn.Module):
    def __init__(self, Block, n, num_classes=1000, tile_size=None):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.tile_size = tile_size

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.convs = self._make_layers(n)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * 4, num_classes)

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
                if i > 0 and j == 0:
                    in_chann = chann
                    chann = chann * 2
                    stride = 2
                else:
                    stride = 1

                tile_sz = self.tile_size if i == 0 else None  # Only tile early blocks

                layers.append(
                    self.Block(in_chann, chann, stride, tile_size=tile_sz, use_checkpoint=True)
                )
                in_chann = chann
        return nn.Sequential(*layers)


def ResNet(n, num_classes=1000, tile_size=None):
    return BaseNet(ResBlockA, n, num_classes=num_classes, tile_size=tile_size)


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



def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)):
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        # print("=================================")
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)

        total_loss += loss.item() * data.size(0)
        total_correct += (predicted == target).sum().item()
        total_samples += data.size(0)

        # scheduler.step()
        
    return total_loss / total_samples, total_correct / total_samples


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
                enumerate(data_loader), total=len(data_loader), desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss = F.nll_loss(output, target)
            loss = criterion(output, target)

            _, predicted = torch.max(output.data, 1)

            total_loss += loss.item() * data.size(0)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        epochs: int = 10,
) -> None:
    print("Training...")
    model.to(get_device())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    for epoch in range(epochs):
        nvtx.range_push(f"Train batch {i}")
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, data_loader, optimizer)
        print(f"Epoch {epoch + 1} / {epochs} | " +
              f"Train Loss: {train_loss:.4f} | " +
              f"Train Acc: {train_acc:.4f}")
        scheduler.step()



    print("Training complete!")




import torch
import torch.nn as nn
import torch.optim as optim
import os
from common.utils import get_device
from typing import Tuple
from tqdm import tqdm
import re
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pynvml
import torch.cuda.nvtx as nvtx
import time

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# GPU logging helpers
def log_gpu_stats():
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return alloc, reserved

def log_nvml():
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    mem_used = mem_info.used / (1024 ** 2)
    return util, mem_used

# Remainder of your existing model, data loader, and model definitions remain unchanged...
# ... [omitted for brevity] ...


def train(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        epochs: int = 10,
) -> None:
    print("Training...")
    device = get_device()
    model.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    for epoch in range(epochs):
        torch.cuda.reset_peak_memory_stats()

        nvtx.range_push(f"Train epoch {epoch}")
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, data_loader, optimizer)

        end_time = time.time()
        nvtx.range_pop()
        torch.cuda.synchronize()

        elapsed = end_time - start_time
        alloc, reserved = log_gpu_stats()
        util, mem_used = log_nvml()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        print(f"Epoch {epoch + 1} / {epochs} | " +
              f"Train Loss: {train_loss:.4f} | " +
              f"Train Acc: {train_acc:.4f} | " +
              f"Time: {elapsed:.2f}s | " +
              f"Alloc: {alloc:.2f} MB | Reserved: {reserved:.2f} MB | " +
              f"Util: {util:.2f}% | Mem Used: {mem_used:.2f} MB | Peak Alloc: {peak:.2f} MB")

        scheduler.step()

    print("Training complete!")

def argparse_args():
    import argparse
    parser = argparse.ArgumentParser(description="memopt")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--tile_size", type=int, default=64, help="Tile size for the model")
    return parser.parse_args()

def main():

    args = argparse_args()
    bs = args.batch_size
    ts = args.tile_size

    batch_size = bs
    weights_path = "./weights-imagenet/00-imagenet-resnet-tile-20-bs-{}-ts-{}.pth".format(batch_size, ts)

    train_loader, test_loader = get_data_imagenet("/home/akash/Desktop/ML-2/imagenet-mini", batch_size=batch_size)
    model = ResNet(n=3, num_classes=1000, tile_size=ts)

    print("Model Parameters Count:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    train(model, train_loader, optimizer, epochs=10)

    torch.save(model.state_dict(), weights_path)

    # Inference Logging
    device = get_device()
    model.eval()
    model.to(device)
    times, allocs, reserveds, utils, mem_usages, peaks = [], [], [], [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            torch.cuda.reset_peak_memory_stats()
            nvtx.range_push(f"Inference batch {i}")
            start_time = time.time()

            output = model(data)

            end_time = time.time()
            nvtx.range_pop()
            torch.cuda.synchronize()

            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            elapsed = end_time - start_time
            alloc, reserved = log_gpu_stats()
            util, mem_used = log_nvml()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

            times.append(elapsed)
            allocs.append(alloc)
            reserveds.append(reserved)
            utils.append(util)
            mem_usages.append(mem_used)
            peaks.append(peak)

    accuracy = 100 * correct / total
    misprediction_rate = 100 - accuracy

    print("##########################################################################")
    print("Batch Size:", batch_size)
    print("Tile Size:", ts)
    print("Model:", weights_path)
    print(f"Total Inference Time: {sum(times):.4f} sec")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Misprediction Rate: {misprediction_rate:.2f}%")
    print(f"Average Inference Time: {sum(times) / len(times):.4f} sec")
    print(f"Average Memory Allocated: {sum(allocs) / len(allocs):.2f} MB")
    print(f"Average Memory Reserved: {sum(reserveds) / len(reserveds):.2f} MB")
    print(f"Average GPU Utilization: {sum(utils) / len(utils):.2f}%")
    print(f"Average Memory Usage: {sum(mem_usages) / len(mem_usages):.2f} MB")
    print(f"Average Peak Memory Allocated: {sum(peaks) / len(peaks):.2f} MB")
    print("##########################################################################")
    print()
    
if __name__ == "__main__":
    main()
    
