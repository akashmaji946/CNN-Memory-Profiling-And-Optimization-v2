import re
import matplotlib.pyplot as plt

# Define your log files and corresponding learning rates
log_files = {
    "001-c.txt": "0.1",
    "001-b.txt": "0.01",
    "001-a.txt": "0.001",
    "001.txt": "0.0001"
}

def parse_log_file(filepath):
    epochs, losses, accs = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r"Epoch\s+(\d+)\s+/.*?\|\s+Train Loss:\s+([\d.]+)\s+\|\s+Train Acc:\s+([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accs.append(float(match.group(3)))
    return epochs, losses, accs

# Initialize plot
plt.figure(figsize=(14, 6))

# Plot Training Loss
plt.subplot(1, 2, 1)
for filename, lr in log_files.items():
    epochs, losses, _ = parse_log_file(filename)
    plt.plot(epochs, losses, label=f"LR = {lr}", marker='o')
plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot Training Accuracy
plt.subplot(1, 2, 2)
for filename, lr in log_files.items():
    epochs, _, accs = parse_log_file(filename)
    plt.plot(epochs, accs, label=f"LR = {lr}", marker='o')
plt.title("Training Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
