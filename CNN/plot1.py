import re
import matplotlib.pyplot as plt

# Load profiling log
with open('profiling_log.txt', 'r') as f:
    log = f.read()

# Regex to extract batch size blocks
batch_pattern = re.compile(r"Batch Size: (\d+).*?Total Inference Time: ([\d.]+) sec.*?Average Inference Time: ([\d.]+) sec.*?Average GPU Utilization: ([\d.]+)%.*?Average Memory Usage: ([\d.]+) MB.*?Average Peak Memory Allocated: ([\d.]+) MB", re.DOTALL)

# Extract data
data = []
for match in batch_pattern.finditer(log):
    batch_size = int(match.group(1))
    total_time = float(match.group(2))
    avg_time = float(match.group(3))
    gpu_util = float(match.group(4))
    mem_usage = float(match.group(5))
    peak_mem = float(match.group(6))
    data.append((batch_size, total_time, avg_time, gpu_util, mem_usage, peak_mem))

# Sort by batch size
data.sort(key=lambda x: x[0])

# Unpack
batch_sizes, total_times, avg_times, gpu_utils, mem_usages, peak_mems = zip(*data)

# Plotting
def plot_metric(x, y, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Generate plots
plot_metric(batch_sizes, total_times, "Seconds", "Total Inference Time vs Batch Size (ResNet-20)", "total_inference_time.png")
plot_metric(batch_sizes, avg_times, "Seconds", "Average Inference Time vs Batch Size (ResNet-20)", "avg_inference_time.png")
plot_metric(batch_sizes, gpu_utils, "GPU Utilization (%)", "GPU Utilization vs Batch Size (ResNet-20)", "gpu_utilization.png")
plot_metric(batch_sizes, mem_usages, "Memory (MB)", "Average Memory Usage vs Batch Size (ResNet-20)", "memory_usage.png")
plot_metric(batch_sizes, peak_mems, "Memory (MB)", "Peak Memory Allocated vs Batch Size (ResNet-20)", "peak_memory.png")
