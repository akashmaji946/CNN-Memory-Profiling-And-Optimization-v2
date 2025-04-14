import matplotlib.pyplot as plt
import numpy as np

models = ["ResNet-20", "ResNet-32", "ResNet-44", "ResNet-56"]
x = np.arange(len(models))
width = 0.35

# ----------------------------
# Metric Data
# ----------------------------

# Inference Time (ms)
infer3_time = [3.74, 5.7, 6.5, 7.6]
infer4_time = [4.3, 5.7, 9.1, 10.9]

# Avg Memory Allocated (MB)
infer3_alloc = [20.98, 21.75, 22.53, 23.30]
infer4_alloc = [20.96, 21.73, 22.51, 23.28]

# Peak Memory Allocated (MB)
infer3_peak = [341.26, 342.03, 342.81, 343.58]
infer4_peak = [181.27, 182.06, 182.86, 183.65]

# GPU Utilization (%)
infer3_util = [16.0, 13.6, 27.6, 20.0]
infer4_util = [11.1, 19.3, 16.6, 20.2]

# ----------------------------
# Plotting Utility Function
# ----------------------------
def plot_metric(y1, y2, ylabel, title, color1='steelblue', color2='seagreen', ylimit=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, y1, width, label='Without optimization (FP32)', color=color1)
    rects2 = ax.bar(x + width/2, y2, width, label='With optimization(AMP + AMC)', color=color2)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    if ylimit: ax.set_ylim(*ylimit)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

# ----------------------------
# Generate Plots
# ----------------------------

plot_metric(infer3_time, infer4_time, "Avg Inference Time (ms)", "Average Inference Time per Batch (size=1024)")

plot_metric(infer3_alloc, infer4_alloc, "Avg Memory Allocated (MB)", "Average Memory Allocated")

plot_metric(infer3_peak, infer4_peak, "Peak Memory Allocated (MB)", "Peak Memory Allocated")

plot_metric(infer3_util, infer4_util, "GPU Utilization (%)", "Average GPU Utilization", ylimit=(0, 35))
