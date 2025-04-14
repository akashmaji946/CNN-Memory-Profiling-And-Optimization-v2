import re
import matplotlib.pyplot as plt

log_files = {
    "ResNet-20": "resnet20.txt",
    "ResNet-32": "resnet32.txt",
    "ResNet-44": "resnet44.txt",
    "ResNet-56": "resnet56.txt"
}

# Store parsed data
train_data = {}
test_data = {}

# Patterns
epoch_pattern = re.compile(r"Epoch (\d+) / \d+ \| Train Loss: ([\d.]+) \| Train Acc: ([\d.]+)")
test_pattern = re.compile(r"Test Loss: ([\d.]+) \| Test Acc: ([\d.]+)")

for model, filepath in log_files.items():
    with open(filepath, "r") as f:
        content = f.read()

    # Parse training metrics
    epochs = []
    losses = []
    accs = []

    for match in epoch_pattern.finditer(content):
        epoch, loss, acc = match.groups()
        epochs.append(int(epoch))
        losses.append(float(loss))
        accs.append(float(acc))

    train_data[model] = {
        "epochs": epochs,
        "loss": losses,
        "acc": accs
    }

    # Parse final test results
    test_match = test_pattern.search(content)
    if test_match:
        test_loss, test_acc = test_match.groups()
        test_data[model] = {
            "loss": float(test_loss),
            "acc": float(test_acc)
        }

# Plot training loss and accuracy
plt.figure(figsize=(12, 5))
for model, data in train_data.items():
    epochs = data["epochs"]
    plt.plot(epochs, data["loss"], marker='o', label=f'{model} - Train Loss')
    plt.plot(epochs, data["acc"], marker='x', label=f'{model} - Train Acc')

plt.title("Training Loss & Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot test loss and accuracy as bar graphs
models = list(test_data.keys())
test_losses = [test_data[m]["loss"] for m in models]
test_accs = [test_data[m]["acc"] for m in models]

x = range(len(models))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([i - width/2 for i in x], test_losses, width, label='Test Loss')
plt.bar([i + width/2 for i in x], test_accs, width, label='Test Accuracy')

plt.title("Test Loss & Accuracy")
plt.xlabel("Model")
plt.ylabel("Value")
plt.xticks(x, models)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
