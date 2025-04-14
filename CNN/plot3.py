import re
import os
import matplotlib.pyplot as plt

# Define your log files
log_files = {
    "ResNet-110 with fp32": "rs110-32.txt",
    "ResNet-20 with fp16": "rs110-16.txt"
}

# Regex pattern to extract metrics
pattern = re.compile(
    r"Batch Size: (\d+).*?"
    r"Total Inference Time: ([\d.]+) sec.*?"
    r"Average Inference Time: ([\d.]+) sec.*?"
    r"Average GPU Utilization: ([\d.]+)%.*?"
    r"Average Memory Usage: ([\d.]+) MB.*?"
    r"Average Peak Memory Allocated: ([\d.]+) MB",
    re.DOTALL,
)

# Dictionary to hold data per model
metrics = {
    "total_time": {},
    "avg_time": {},
    "gpu_util": {},
    "mem_usage": {},
    "peak_mem": {}
}

# Parse each file
for model_name, filename in log_files.items():
    with open(filename, 'r') as f:
        log = f.read()
    entries = pattern.findall(log)

    # Prepare data lists sorted by batch size
    batch_data = sorted([(int(b), float(t), float(a), float(g), float(m), float(p)) for b, t, a, g, m, p in entries])
    batch_sizes, total_times, avg_times, gpu_utils, mem_usages, peak_mems = zip(*batch_data)

    metrics["total_time"][model_name] = (batch_sizes, total_times)
    metrics["avg_time"][model_name] = (batch_sizes, avg_times)
    metrics["gpu_util"][model_name] = (batch_sizes, gpu_utils)
    metrics["mem_usage"][model_name] = (batch_sizes, mem_usages)
    metrics["peak_mem"][model_name] = (batch_sizes, peak_mems)

# Plotting function
def plot_comparison(metric_key, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for model_name, (x, y) in metrics[metric_key].items():
        plt.plot(x, y, marker='o', label=model_name)
    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


# Plot all metrics
plot_comparison("total_time", "Seconds", "Total Inference Time vs Batch Size", "total_inference_time_all_models.png")
plot_comparison("avg_time", "Seconds", "Average Inference Time vs Batch Size", "avg_inference_time_all_models.png")
plot_comparison("gpu_util", "GPU Utilization (%)", "GPU Utilization vs Batch Size", "gpu_utilization_all_models.png")
plot_comparison("mem_usage", "Memory (MB)", "Average Memory Usage vs Batch Size", "memory_usage_all_models.png")
plot_comparison("peak_mem", "Memory (MB)", "Peak Memory Allocated vs Batch Size", "peak_memory_all_models.png")
