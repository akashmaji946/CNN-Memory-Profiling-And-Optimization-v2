import subprocess

depths = [110]
batch_sizes = [2 ** i for i in range(11)]  # 2^0 to 2^10 => 1 to 1024

for depth in depths:
    weights_path = f"./weights/v9-cifar10-resnet-{depth}.pth"
    for batch_size in batch_sizes:
        print(f"\n>>> Running ResNet-{depth} | Batch Size: {batch_size}")
        subprocess.run([
            "python", "infer-fp.py",
            "--weights", weights_path,
            "--batch_size", str(batch_size)
        ])
        print(f"Finished ResNet-{depth} | Batch Size: {batch_size}")
        print("-----------------------------------------------------")