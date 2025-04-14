# import re
# import os
# import matplotlib.pyplot as plt

# # Define your log files
# log_files = {
#     "ResNet-110 with fp32": "rs110-32.txt",
#     "ResNet-110 with fp16": "rs110-16.txt"
# }

# # Regex patterns
# metric_pattern = re.compile(
#     r"Batch Size: (\d+).*?"
#     r"Total Inference Time: ([\d.]+) sec.*?"
#     r"Average Inference Time: ([\d.]+) sec.*?"
#     r"Average GPU Utilization: ([\d.]+)%.*?"
#     r"Average Memory Usage: ([\d.]+) MB.*?"
#     r"Average Peak Memory Allocated: ([\d.]+) MB",
#     re.DOTALL,
# )

# accuracy_pattern = re.compile(
#     r"Batch Size: (\d+).*?"
#     r"Test Accuracy: ([\d.]+)%.*?"
#     r"Misprediction Rate: ([\d.]+)%",
#     re.DOTALL,
# )

# # Store metrics
# metrics = {
#     "total_time": {},
#     "avg_time": {},
#     "gpu_util": {},
#     "mem_usage": {},
#     "peak_mem": {}
# }

# accuracy_metrics = {
#     "accuracy": {},
#     "loss": {}
# }

# # Parse each file
# for model_name, filename in log_files.items():
#     with open(filename, 'r') as f:
#         log = f.read()

#     # Inference metrics
#     entries = metric_pattern.findall(log)
#     batch_data = sorted([(int(b), float(t), float(a), float(g), float(m), float(p)) for b, t, a, g, m, p in entries])
#     batch_sizes, total_times, avg_times, gpu_utils, mem_usages, peak_mems = zip(*batch_data)

#     metrics["total_time"][model_name] = (batch_sizes, total_times)
#     metrics["avg_time"][model_name] = (batch_sizes, avg_times)
#     metrics["gpu_util"][model_name] = (batch_sizes, gpu_utils)
#     metrics["mem_usage"][model_name] = (batch_sizes, mem_usages)
#     metrics["peak_mem"][model_name] = (batch_sizes, peak_mems)

#     # Accuracy/Loss
#     acc_entries = accuracy_pattern.findall(log)
#     if acc_entries:
#         # Use largest batch size entry (or choose a specific one)
#         batch_size, acc, mis = max(acc_entries, key=lambda x: int(x[0]))
#         accuracy_metrics["accuracy"][model_name] = float(acc)
#         accuracy_metrics["loss"][model_name] = float(mis)


# # Line plot comparison
# def plot_comparison(metric_key, ylabel, title, filename):
#     plt.figure(figsize=(10, 6))
#     for model_name, (x, y) in metrics[metric_key].items():
#         plt.plot(x, y, marker='o', label=model_name)
#     plt.xscale("log", base=2)
#     plt.xticks(sorted(set(x for x_list, _ in metrics[metric_key].values() for x in x_list)))
#     plt.xlabel("Batch Size (log scale)")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()

# # Bar plot for accuracy/loss
# def plot_bar_comparison(data_dict, ylabel, title, filename):
#     plt.figure(figsize=(8, 5))
#     labels = list(data_dict.keys())
#     values = list(data_dict.values())
#     plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
#     for i, val in enumerate(values):
#         plt.text(i, val + 0.5, f"{val:.2f}", ha='center', va='bottom')
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()

# # Generate plots
# plot_comparison("total_time", "Seconds", "Total Inference Time vs Batch Size", "total_inference_time_all_models.png")
# plot_comparison("avg_time", "Seconds", "Average Inference Time vs Batch Size", "avg_inference_time_all_models.png")
# plot_comparison("gpu_util", "GPU Utilization (%)", "GPU Utilization vs Batch Size", "gpu_utilization_all_models.png")
# plot_comparison("mem_usage", "Memory (MB)", "Average Memory Usage vs Batch Size", "memory_usage_all_models.png")
# plot_comparison("peak_mem", "Memory (MB)", "Peak Memory Allocated vs Batch Size", "peak_memory_all_models.png")

# # Accuracy and Loss Bar Charts
# plot_bar_comparison(accuracy_metrics["accuracy"], "Accuracy (%)", "Test Accuracy Comparison", "test_accuracy_comparison.png")
# plot_bar_comparison(accuracy_metrics["loss"], "Misprediction Rate (%)", "Test Loss (Misprediction) Comparison", "test_loss_comparison.png")

import re
import os
import numpy as np
import matplotlib.pyplot as plt

# Define log files
log_files = {
    "ResNet-110 with fp32": "rs110-32.txt",
    "ResNet-110 with fp16": "rs110-16.txt"
}

# Regex patterns
metric_pattern = re.compile(
    r"Batch Size: (\d+).*?"
    r"Total Inference Time: ([\d.]+) sec.*?"
    r"Average Inference Time: ([\d.]+) sec.*?"
    r"Average GPU Utilization: ([\d.]+)%.*?"
    r"Average Memory Usage: ([\d.]+) MB.*?"
    r"Average Peak Memory Allocated: ([\d.]+) MB",
    re.DOTALL,
)

accuracy_pattern = re.compile(
    r"Batch Size: (\d+).*?"
    r"Test Accuracy: ([\d.]+)%.*?"
    r"Misprediction Rate: ([\d.]+)%",
    re.DOTALL,
)

# Metrics dictionaries
metrics = {
    "total_time": {},
    "avg_time": {},
    "gpu_util": {},
    "mem_usage": {},
    "peak_mem": {}
}

# Accuracy/loss per batch size
batchwise_accuracy = {}  # {model_name: {batch_size: accuracy}}
batchwise_loss = {}      # {model_name: {batch_size: loss}}

# Parse logs
for model_name, filename in log_files.items():
    with open(filename, 'r') as f:
        log = f.read()

    # Inference metrics
    entries = metric_pattern.findall(log)
    batch_data = sorted([(int(b), float(t), float(a), float(g), float(m), float(p)) for b, t, a, g, m, p in entries])
    batch_sizes, total_times, avg_times, gpu_utils, mem_usages, peak_mems = zip(*batch_data)

    metrics["total_time"][model_name] = (batch_sizes, total_times)
    metrics["avg_time"][model_name] = (batch_sizes, avg_times)
    metrics["gpu_util"][model_name] = (batch_sizes, gpu_utils)
    metrics["mem_usage"][model_name] = (batch_sizes, mem_usages)
    metrics["peak_mem"][model_name] = (batch_sizes, peak_mems)

    # Accuracy/loss per batch size
    acc_entries = accuracy_pattern.findall(log)
    for batch_size, acc, mis in acc_entries:
        batch_size = int(batch_size)
        batchwise_accuracy.setdefault(model_name, {})[batch_size] = float(acc)
        batchwise_loss.setdefault(model_name, {})[batch_size] = float(mis)

# Line plot for inference metrics
def plot_comparison(metric_key, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for model_name, (x, y) in metrics[metric_key].items():
        plt.plot(x, y, marker='o', label=model_name)
    plt.xscale("log", base=2)
    plt.xticks(sorted(set(x for x_list, _ in metrics[metric_key].values() for x in x_list)))
    plt.xlabel("Batch Size (log scale)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Grouped bar plots for accuracy and loss
def plot_grouped_bar(data_dict, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    models = list(data_dict.keys())
    all_batches = sorted(set(b for m in models for b in data_dict[m].keys()))
    x = np.arange(len(all_batches))
    width = 0.35

    for i, model in enumerate(models):
        values = [data_dict[model].get(b, 0) for b in all_batches]
        plt.bar(x + i * width, values, width=width, label=model)

    plt.xlabel("Batch Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x + width / 2, all_batches)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Generate all plots
plot_comparison("total_time", "Seconds", "Total Inference Time vs Batch Size", "total_inference_time_all_models.png")
plot_comparison("avg_time", "Seconds", "Average Inference Time vs Batch Size", "avg_inference_time_all_models.png")
plot_comparison("gpu_util", "GPU Utilization (%)", "GPU Utilization vs Batch Size", "gpu_utilization_all_models.png")
plot_comparison("mem_usage", "Memory (MB)", "Average Memory Usage vs Batch Size", "memory_usage_all_models.png")
plot_comparison("peak_mem", "Memory (MB)", "Peak Memory Allocated vs Batch Size", "peak_memory_all_models.png")

plot_grouped_bar(batchwise_accuracy, "Accuracy (%)", "Test Accuracy per Batch Size", "test_accuracy_per_batch.png")
plot_grouped_bar(batchwise_loss, "Misprediction Rate (%)", "Test Loss per Batch Size", "test_loss_per_batch.png")
