import matplotlib.pyplot as plt

# Tile sizes
tile_sizes = [8, 16, 32, 64]

# Metrics
inference_time = [29.715, 7.082, 2.632, 1.022]         # In seconds
gpu_utilization = [49.55, 90.37, 97.13, 98.77]         # In %
gpu_memory = [855.12, 865.12, 881.12 , 889.12]          # Average GPU memory in MB

# Create subplots
fig, ax1 = plt.subplots(figsize=(12, 6))

# Inference time plot
color1 = 'tab:red'
ax1.set_xlabel('Tile Size')
ax1.set_ylabel('Inference Time (s)', color=color1)
ax1.plot(tile_sizes, inference_time, marker='o', color=color1, label='Inference Time')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(tile_sizes)

# GPU utilization on second Y axis
ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('GPU Utilization (%)', color=color2)
ax2.plot(tile_sizes, gpu_utilization, marker='s', linestyle='--', color=color2, label='GPU Utilization')
ax2.tick_params(axis='y', labelcolor=color2)

# Create a third axis for memory usage
ax3 = ax1.twinx()
color3 = 'tab:green'
ax3.spines["right"].set_position(("outward", 60))  # Offset third y-axis
ax3.set_ylabel('Avg GPU Memory (MB)', color=color3)
ax3.plot(tile_sizes, gpu_memory, marker='^', linestyle='-.', color=color3, label='GPU Memory')
ax3.tick_params(axis='y', labelcolor=color3)

# Title and layout
plt.title('Tile Size vs [Inference Time, GPU Utilization, and Average Memory Usage]')
fig.tight_layout()
plt.grid(True)
plt.show()
