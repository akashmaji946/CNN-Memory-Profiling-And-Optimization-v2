
import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import *
from common.train_utils import *

def extract_tile_with_overlap(x, top, left, tile_size, padding):
    B, C, H, W = x.shape
    top_pad = max(padding - top, 0)
    left_pad = max(padding - left, 0)
    bottom_pad = max(padding - (H - (top + tile_size)), 0)
    right_pad = max(padding - (W - (left + tile_size)), 0)

    # Determine crop bounds (with overlap)
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
            # Extract tile with overlap
            tile = extract_tile_with_overlap(x, top, left, tile_size, padding)

            # Run convolution on the tile
            tile_out = conv_layer(tile)

            # Determine the valid output region
            h_out = min(tile_size, H - top)
            w_out = min(tile_size, W - left)

            # Compute padding adjustment for the tile_out (since it may include padding)
            top_pad = max(padding - top, 0)
            left_pad = max(padding - left, 0)

            # Ensure the output dimensions are properly aligned with tile_out dimensions
            h_out_conv = (h_out + 2 * padding - kernel_size) // stride + 1
            w_out_conv = (w_out + 2 * padding - kernel_size) // stride + 1

            # Adjust the slice indices for the output tensor
            output_slice = output[:, :, top:top + h_out_conv, left:left + w_out_conv]
            tile_out_slice = tile_out[:, :, top_pad:top_pad + h_out_conv, left_pad:left_pad + w_out_conv]

            # Ensure that the dimensions of output_slice and tile_out_slice match
            assert output_slice.shape == tile_out_slice.shape, f"Shape mismatch: {output_slice.shape} vs {tile_out_slice.shape}"

            output_slice.copy_(tile_out_slice)  # Copy the tile_out slice into the output slice

    return output


class ResBlockA(nn.Module):
    def __init__(self, in_chann, chann, stride, tile_size=None):  # Add tile_size as an optional argument
        super(ResBlockA, self).__init__()

        self.tile_size = tile_size  # Store tile_size if provided

        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)

        # A 1x1 convolution to match the dimensions if the input and output shapes do not match
        self.match_dimensions = nn.Conv2d(in_chann, chann, kernel_size=1, stride=stride, padding=0) if stride != 1 else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.functional.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y)
        
        # If the input and output dimensions do not match, adjust 'z' accordingly
        if x.shape != y.shape:
            if self.match_dimensions:
                z = self.match_dimensions(x)  # Apply the 1x1 convolution to match dimensions
            else:
                z = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
                x_channel = x.size(1)
                y_channel = y.size(1)
                ch_res = (y_channel - x_channel) // 2
                pad = (0, 0, 0, 0, ch_res, ch_res)
                z = nn.functional.pad(z, pad=pad, mode="constant", value=0)
        else:
            z = x

        z = z + y
        z = nn.functional.relu(z)
        return z


class PlainBlock(nn.Module):

    def __init__(self, in_chann, chann, stride, tile_size):
        super(PlainBlock, self).__init__()

        self.tile_size = tile_size

        # Original convolution layers
        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1   = nn.BatchNorm2d(chann)
        
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(chann)

    def forward(self, x):
        y = tiled_conv2d(x, self.conv1, self.tile_size)  # Use tiled convolution
        y = self.bn1(y)
        y = nn.functional.relu(y)
        
        y = tiled_conv2d(y, self.conv2, self.tile_size)  # Use tiled convolution
        y = self.bn2(y)
        y = nn.functional.relu(y)
        return y


class BaseNet(nn.Module):
    def __init__(self, Block, n, tile_size):
        super(BaseNet, self).__init__()
        self.Block = Block
        self.tile_size = tile_size  # Store the tile_size parameter
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

                layers += [self.Block(in_chann, chann, stride, self.tile_size)]  # Pass tile_size to the block

                stride = 1
                in_chann = chann

        return nn.Sequential(*layers)


def ResNet(n, tile_size):
    return BaseNet(ResBlockA, n, tile_size)

def PlainNet(n, tile_size):
    return BaseNet(PlainBlock, n, tile_size)



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
    model = ResNet(n=3, tile_size=8)  # Specify the tile size here
    print("Model Parameters Count :", sum(p.numel() for p in model.parameters()))

    # Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, optimizer, epochs=50)

    # will save the model
    # torch.save(model.state_dict(), "./weights/v9-cifar10.pth")
    torch.save(model.state_dict(), "./weights/v9-cifar10-resnet-tile-20.pth")

    # Evaluate the model
    test_loss, test_acc = evaluate(model, test_loader)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    main()
