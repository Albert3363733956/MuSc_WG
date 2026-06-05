import os
from pathlib import Path
import subprocess
import sys

# Configuration
device = "0"
project_root = Path(__file__).resolve().parents[1]
output_root = (project_root / ".." / ".." / "output" / "AdaCLIP").resolve()
default_checkpoint = project_root / "weights" / "pretrained_visa_clinicdb.pth"

# Paths to datasets
data_root_btad = Path(r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed")
data_root_mvtec_loco = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec_loco")

# Define test configurations
# Set "class_name" to a concrete class, such as "01" or "breakfast_box", to test a single class.
test_configs = [
    {"dataset": "btad", "path": data_root_btad, "class_name": "all", "checkpoint": default_checkpoint},
    {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all", "checkpoint": default_checkpoint},
]

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = Path(config["path"])
    target_class = config["class_name"]
    checkpoint_path = Path(config["checkpoint"])

    # Check if data root exists
    if not data_root.exists():
        print(f"Error: Data root not found at {data_root.resolve()} for dataset {test_dataset}")
        continue

    meta_path = data_root / "meta.json"
    if not meta_path.exists():
        print(f"Error: meta.json not found at {meta_path} for dataset {test_dataset}")
        continue

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path} for dataset {test_dataset}")
        continue

    # Paths
    if target_class.lower() == "all":
        save_dir = output_root / test_dataset / "zero_shot_all"
    else:
        save_dir = output_root / test_dataset / f"zero_shot_{target_class}"
    
    # Ensure save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Construct the command
    cmd = [
        sys.executable, "test.py",
        "--testing_data", test_dataset,
        "--data_root", str(data_root),
        "--ckt_path", str(checkpoint_path),
        "--save_path", str(save_dir),
        "--obj_name", target_class,
        "--testing_model", "dataset",
        "--save_fig", "True"
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running test for dataset={test_dataset}, class={target_class}...")
    
    try:
        subprocess.run(cmd, env=env, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
