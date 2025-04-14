import os
import subprocess

# Path to weights
model_paths = [
    "./weights/v9-cifar10-resnet-20.pth",
    "./weights/v9-cifar10-resnet-32.pth",
    "./weights/v9-cifar10-resnet-44.pth",
    "./weights/v9-cifar10-resnet-56.pth",
    "./weights/v9-cifar10-resnet-110.pth",
]

# Batch sizes to test
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Path to the inference script
inference_script = "infer.py"  # Change if named differently

# Output directories
os.makedirs("profiles", exist_ok=True)
os.makedirs("stats", exist_ok=True)

for model_path in model_paths:
    model_name = os.path.basename(model_path).replace(".pth", "")

    for batch_size in batch_sizes:
        trace_file = f"profiles/{model_name}_bs{batch_size}"
        stats_file = f"stats/{model_name}_bs{batch_size}.txt"

        print(f"> Profiling {model_name} with batch size {batch_size}...")

        # Run nsys profile
        subprocess.run([
            "nsys", "profile",
            "--trace=cuda",
            "-o", trace_file,
            "python", inference_script,
            "--weights", model_path,
            "--batch_size", str(batch_size)
        ])

        subprocess.run(f"nsys stats {trace_file}.nsys-rep > {stats_file}", shell=True)

print("all profiling complete.")
