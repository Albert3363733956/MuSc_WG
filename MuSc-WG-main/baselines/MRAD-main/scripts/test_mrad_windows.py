import os
import subprocess
import sys
import importlib.util
from pathlib import Path

# Configuration
device = "0"
project_root = Path(__file__).resolve().parents[1]
output_root = project_root.parents[1] / "output" / "MRAD"


def ensure_meta_file(dataset_name, data_root):
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        return True

    preprocessors = {
        "hhled": (project_root / "generate_dataset_json" / "hhled.py", "HHLEDSolver"),
        "hhled_ad": (project_root / "generate_dataset_json" / "hhled.py", "HHLEDSolver"),
    }
    if dataset_name not in preprocessors:
        print(f"Error: meta.json not found at {meta_path.resolve()} for dataset {dataset_name}")
        return False

    module_path, class_name = preprocessors[dataset_name]
    spec = importlib.util.spec_from_file_location(f"{dataset_name}_solver", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver_cls = getattr(module, class_name)
    print(f"meta.json not found at {meta_path.resolve()}; generating it for dataset {dataset_name}...")
    solver_cls(root=data_root).run()
    return meta_path.exists()

# Path to datasets
data_root_mvtec = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec")
data_root_mvtec_loco = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec_loco")
data_root_microled = Path(r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD")
data_root_miniled = Path(r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD")
data_root_hhled = Path(r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD")

default_checkpoint = project_root / "checkpoints" / "test_on_mvtec.pth"

# Define test configurations
# Uncomment the configuration you want to run
test_configs = [
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all", "checkpoint": default_checkpoint},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "breakfast_box", "checkpoint": default_checkpoint},
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "bottle", "checkpoint": default_checkpoint},
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all", "checkpoint": default_checkpoint},
    {"dataset": "microled", "path": data_root_microled, "class_name": "all", "checkpoint": default_checkpoint},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all", "checkpoint": default_checkpoint},
    # {"dataset": "hhled", "path": data_root_hhled, "class_name": "all", "checkpoint": default_checkpoint},
]

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = Path(config["path"])
    target_class = config["class_name"]
    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    # Check if data root exists
    if not data_root.exists():
        print(f"Error: Data root not found at {data_root.resolve()} for dataset {test_dataset}")
        continue
    if not ensure_meta_file(test_dataset, data_root):
        continue
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path.resolve()} for dataset {test_dataset}")
        continue

    # Paths
    if target_class.lower() == "all":
        save_dir = output_root / test_dataset / "zero_shot_all"
    else:
        save_dir = output_root / test_dataset / f"zero_shot_{target_class}"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the command
    cmd = [
        sys.executable, str(project_root / "test.py"),
        "--dataset", test_dataset,
        "--data_path", str(data_root),
        "--checkpoint_path", str(checkpoint_path),
        "--cache_dir", str(project_root / "cache"),
        "--save_path", str(save_dir),
        "--obj_name", target_class,
        "--model_type", "mrad-clip",
        "--metrics", "image-pixel-level",
        "--visulize_bool", "True"
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running test for dataset={test_dataset}, class={target_class}...")
    
    try:
        subprocess.run(cmd, env=env, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
