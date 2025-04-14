import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from common.utils import get_device
from typing import Tuple
from tqdm import tqdm


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
        train_loss, train_acc = train_epoch(model, data_loader, optimizer)
        print(f"Epoch {epoch + 1} / {epochs} | " +
              f"Train Loss: {train_loss:.4f} | " +
              f"Train Acc: {train_acc:.4f}")
        scheduler.step()
    print("Training complete!")
