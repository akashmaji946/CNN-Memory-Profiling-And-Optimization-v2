import torch
import torch.nn as nn
import torch.optim as optim
import os
from common.utils import *
from common.train_utils import *
import torch
import torch.nn.functional as F

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

def tiled_conv2d(x, conv_layer, tile_size):
    B, C, H, W = x.shape
    padding = conv_layer.padding[0]  # assumes square padding
    stride = conv_layer.stride[0]
    kernel_size = conv_layer.kernel_size[0]

    out_channels = conv_layer.out_channels
    output = torch.zeros(B, out_channels, H, W, device=x.device)

    for top in range(0, H, tile_size):
        for left in range(0, W, tile_size):
            tile = extract_tile_with_overlap(x, top, left, tile_size, padding)
            tile_out = conv_layer(tile)

            h_out = min(tile_size, H - top)
            w_out = min(tile_size, W - left)

            top_pad = max(padding - top, 0)
            left_pad = max(padding - left, 0)

            h_out_conv = (h_out + 2 * padding - kernel_size) // stride + 1
            w_out_conv = (w_out + 2 * padding - kernel_size) // stride + 1

            output_slice = output[:, :, top:top + h_out_conv, left:left + w_out_conv]
            tile_out_slice = tile_out[:, :, top_pad:top_pad + h_out_conv, left_pad:left_pad + w_out_conv]

            assert output_slice.shape == tile_out_slice.shape, f"Shape mismatch: {output_slice.shape} vs {tile_out_slice.shape}"

            output_slice.copy_(tile_out_slice)

    return output

class ResBlockA(nn.Module):
    def __init__(self, in_chann, chann, stride, tile_size=None):
        super(ResBlockA, self).__init__()
        self.tile_size = tile_size

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

        # Use tiled conv if tile_size is provided
        if self.tile_size:
            out = tiled_conv2d(x, self.conv1, self.tile_size)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = nn.functional.relu(out)

        if self.tile_size:
            out = tiled_conv2d(out, self.conv2, self.tile_size)
        else:
            out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out


class BaseNet(nn.Module):
    def __init__(self, Block, n, num_classes=1000, tile_size=None):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.tile_size = tile_size

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # no tiling
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

                layers.append(self.Block(in_chann, chann, stride, tile_size=self.tile_size))
                in_chann = chann
        return nn.Sequential(*layers)


def ResNet(n, num_classes=1000, tile_size=None):
    return BaseNet(ResBlockA, n, num_classes=num_classes, tile_size=tile_size)


transform_func = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize([125., 123., 114.], [1., 1., 1.])
        ])


import argparse
import torch
import torch.nn.functional as F

from common.utils import get_device, get_data
# from main import ResNet, transform_func  # make sure this is correct for your structure
import re
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()

def extract_depth_from_filename(weights_path):
    match = re.search(r"resnet-tile-(\d+)", weights_path)
    if match:
        depth = int(match.group(1))
        if (depth - 2) % 6 != 0:
            raise ValueError(f"Invalid ResNet depth in file: {weights_path}")
        return (depth - 2) // 6  # Compute n
    else:
        raise ValueError(f"Could not parse depth from file name: {weights_path}")


import pynvml
import torch.cuda.nvtx as nvtx
import time

# Function to log GPU stats
def log_gpu_stats(label=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return allocated, reserved

# Function to setup NVML
def setup_nvml():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return handle

# Function to log NVML stats
def log_nvml(handle, label=""):
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu, mem_info.used / (1024 ** 2)

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


def run_inference3(weights_path, batch_size):
    device = get_device()
    n = extract_depth_from_filename(weights_path)
    # Load ResNet32 model
    print("##########################################################################")
    print("With tile size => ", 16)
    model = ResNet(n, num_classes=1000, tile_size=16)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    # Get data
    _, test_loader = get_data_imagenet("/home/akash/Desktop/ML-2/imagenet-mini", batch_size=batch_size)

    # Setup NVML for GPU stats
    handle = setup_nvml()
    torch.cuda.reset_peak_memory_stats()

    # Track correct predictions and total
    correct = 0
    total = 0

    # Store data for averages
    times, allocs, reserveds, utils, mem_usages, peaks = [], [], [], [], [], []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            nvtx.range_push(f"Inference batch {i}")
            start_time = time.time()

            output = model(data)

            end_time = time.time()
            nvtx.range_pop()

            torch.cuda.synchronize()

            # Get predictions and compute accuracy
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            # Track peak memory usage
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

            # Log stats
            elapsed = end_time - start_time
            alloc, reserved = log_gpu_stats(f"Batch {i}")
            util, mem_used = log_nvml(handle, f"Batch {i}")

            # print(f"[Batch {i}] Batch Size: {data.size(0)}")
            # print(f"[Batch {i}] Inference time: {elapsed:.4f} sec")
            # print(f"[Batch {i}] Peak Memory Allocated: {peak_mem:.2f} MB\n")

            # Store for averaging
            times.append(elapsed)
            allocs.append(alloc)
            reserveds.append(reserved)
            utils.append(util)
            mem_usages.append(mem_used)
            peaks.append(peak_mem)

    # Calculate overall accuracy and misprediction rate
    accuracy = 100 * correct / total
    misprediction_rate = 100 - accuracy

    print("##########################################################################")
    print("Batch Size:", batch_size)
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

if __name__ == "__main__":
    args = parse_args()  # Make sure parse_args() is set up to parse weights and batch_size
    # run_inference3(args.weights, args.batch_size)  # Run inference with custom ResNet32
    run_inference3(args.weights, args.batch_size)  # Run inference with torchvision ResNet18
    print("##########################################################################")




