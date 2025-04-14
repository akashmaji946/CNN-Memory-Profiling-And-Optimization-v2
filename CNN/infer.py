
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
    match = re.search(r"resnet-(\d+)", weights_path)
    if match:
        depth = int(match.group(1))
        if (depth - 2) % 6 != 0:
            raise ValueError(f"Invalid ResNet depth in file: {weights_path}")
        return (depth - 2) // 6  # Compute n
    else:
        raise ValueError(f"Could not parse depth from file name: {weights_path}")

def run_inference(weights_path, batch_size):
    device = get_device()
    n = extract_depth_from_filename(weights_path)
    model = ResNet(n)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    _, test_loader = get_data("cifar10", batch_size=batch_size, transform=transform_func)

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            preds = torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1)
            # Suppress output during profiling
            pass

import pynvml
import torch.cuda.nvtx as nvtx
import time

# def log_gpu_stats(label=""):
#     allocated = torch.cuda.memory_allocated() / (1024 ** 2)
#     reserved = torch.cuda.memory_reserved() / (1024 ** 2)
#     print(f"[{label}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

# def setup_nvml():
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     return handle

# def log_nvml(handle, label=""):
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     util = pynvml.nvmlDeviceGetUtilizationRates(handle)
#     print(f"[{label}] GPU Utilization: {util.gpu}% | Mem Used: {mem_info.used / (1024 ** 2):.2f} MB")

def run_inference2(weights_path, batch_size):
    device = get_device()
    n = extract_depth_from_filename(weights_path)
    model = ResNet(n)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    _, test_loader = get_data("cifar10", batch_size=batch_size, transform=transform_func)

    handle = setup_nvml()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)

            nvtx.range_push(f"Inference batch {i}")
            start_time = time.time()

            output = model(data)

            end_time = time.time()
            nvtx.range_pop()

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

            print(f"[Batch {i}] Inference time: {end_time - start_time:.4f} sec")
            log_gpu_stats(f"Batch {i}")
            log_nvml(handle, f"Batch {i}")
            print(f"[Batch {i}] Peak Memory Allocated: {peak_mem:.2f} MB")
            print()
            # Optional: break early if you only want a few batches for profiling
            # if i == 5: break


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

def run_inference3(weights_path, batch_size):
    device = get_device()
    n = extract_depth_from_filename(weights_path)
    # Load ResNet32 model
    model = ResNet(n)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.eval()

    # Get data
    _, test_loader = get_data("cifar10", batch_size=batch_size, transform=transform_func)

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

def run_inference4(weights_path, batch_size):
    device = get_device()
    torch.backends.cudnn.benchmark = True  # Enables best algorithm selection

    n = extract_depth_from_filename(weights_path)
    model = ResNet(n)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    model.to(memory_format=torch.channels_last)  # Set model to channels_last
    model.eval()

    _, test_loader = get_data("cifar10", batch_size=batch_size, transform=transform_func)

    handle = setup_nvml()
    torch.cuda.reset_peak_memory_stats()

    correct = 0
    total = 0

    times, allocs, reserveds, utils, mem_usages, peaks = [], [], [], [], [], []

    with torch.inference_mode():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device).to(memory_format=torch.channels_last)  
            target = target.to(device)

            nvtx.range_push(f"Inference batch {i}")
            start_time = time.time()

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                output = model(data)

            end_time = time.time()
            nvtx.range_pop()
            torch.cuda.synchronize()

            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            elapsed = end_time - start_time
            alloc, reserved = log_gpu_stats()
            util, mem_used = log_nvml(handle)

            times.append(elapsed)
            allocs.append(alloc)
            reserveds.append(reserved)
            utils.append(util)
            mem_usages.append(mem_used)
            peaks.append(peak_mem)

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
    print("Infer4")
    run_inference4(args.weights, args.batch_size)  # Run inference with torchvision ResNet18
    print("##########################################################################")



