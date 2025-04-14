import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_tile_with_overlap(x, top, left, tile_size, padding):
    """Extract a tile from x with overlap padding"""
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

            # Run conv
            tile_out = conv_layer(tile)

            # Determine valid region (exclude padding area in output)
            h_out = min(tile_size, H - top)
            w_out = min(tile_size, W - left)

            # Offset to skip padded border in tile_out
            top_pad = max(padding - top, 0)
            left_pad = max(padding - left, 0)

            output[:, :, top:top + h_out, left:left + w_out] = tile_out[:, :, top_pad:top_pad + h_out, left_pad:left_pad + w_out]
    return output

# Sample image
x = torch.randn(1, 3, 32, 32).cuda()

# Simple conv layer
conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1).cuda()

# Tiled output
tiled_out = tiled_conv2d(x, conv, tile_size=16)

# Reference output
ref_out = conv(x)

print(ref_out)  # Should be (1, 8, 32, 32)
print(tiled_out)

# Check correctness
print(torch.allclose(tiled_out, ref_out, atol=1e-2))  # Should be True

