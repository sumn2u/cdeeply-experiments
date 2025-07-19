import subprocess
import itertools

# Define parameter grid
activations = ['relu', 'tanh', 'sigmoid']
layers = [2, 3, 4]
neurons = [32, 64]
quantize_options = [False, True]

# Your dataset path and target index (adjust as needed)
file_path = "seoul_weather.txt"
target_indices = "22"

# Loop through combinations
for activation, layer, neuron, quantize in itertools.product(activations, layers, neurons, quantize_options):
    cmd = [
        "python", "predictions-task.py",  # replace with your script name
        "--file", file_path,
        "--targets", target_indices,
        "--activation", activation,
        "--max_layers", str(layer),
        "--max_neurons", str(neuron),
        "--save_csv"
    ]
    if quantize:
        cmd.append("--quantize")

    print(f"\n🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd)
