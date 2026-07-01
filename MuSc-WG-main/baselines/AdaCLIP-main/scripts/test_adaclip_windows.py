import os
from pathlib import Path
import subprocess
import sys

# Configuration
device = "0"
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
output_root = (project_root / ".." / ".." / "output" / "AdaCLIP").resolve()
default_checkpoint = project_root / "weights" / "pretrained_visa_clinicdb.pth"


def resolve_dataset_root(env_name, *candidates):
    env_value = os.environ.get(env_name)
    paths = [Path(env_value)] if env_value else []
    paths.extend(Path(candidate) for candidate in candidates)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def ensure_meta_file(dataset_name, data_root):
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        return True

    preprocessors = {
        "mvtec": ("data_preprocess.mvtec", "MVTecSolver"),
        "microled": ("data_preprocess.microled", "MicroLEDSolver"),
        "microled_ad": ("data_preprocess.microled", "MicroLEDSolver"),
        "miniled": ("data_preprocess.miniled", "MiniLEDSolver"),
        "miniled_ad": ("data_preprocess.miniled", "MiniLEDSolver"),
        "hhled": ("data_preprocess.hhled", "HHLEDSolver"),
        "hhled_ad": ("data_preprocess.hhled", "HHLEDSolver"),
    }
    if dataset_name not in preprocessors:
        print(f"Error: meta.json not found at {meta_path} for dataset {dataset_name}")
        return False

    module_name, class_name = preprocessors[dataset_name]
    module = __import__(module_name, fromlist=[class_name])
    solver_cls = getattr(module, class_name)
    print(f"meta.json not found at {meta_path}; generating it for dataset {dataset_name}...")
    solver_cls(root=data_root).run()
    return meta_path.exists()


# Paths to datasets
data_root_mvtec = resolve_dataset_root(
    "ADACLIP_MVTEC_ROOT",
    r"C:\Users\Administrator\Desktop\dataset\MVTec",
    project_root.parents[1] / "data" / "mvtec_AD",
    project_root.parents[1] / "data" / "mvtec",
    project_root.parents[1] / "datasets" / "mvtec_AD",
    project_root.parents[1] / "datasets" / "mvtec",
)
data_root_btad = Path(r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed")
data_root_mvtec_loco = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec_loco")
data_root_microled = resolve_dataset_root(
    "ADACLIP_MICROLED_ROOT",
    r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD",
    project_root.parents[1] / "data" / "microled_AD",
    project_root.parents[1] / "datasets" / "microled_AD",
)
data_root_miniled = resolve_dataset_root(
    "ADACLIP_MINILED_ROOT",
    r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD",
    project_root.parents[1] / "data" / "miniled_AD",
    project_root.parents[1] / "datasets" / "miniled_AD",
)
data_root_hhled = resolve_dataset_root(
    "ADACLIP_HHLED_ROOT",
    r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD",
    project_root.parents[1] / "data" / "hhled_AD",
    project_root.parents[1] / "datasets" / "hhled_AD",
)

# Define test configurations
# Set "class_name" to a concrete class, such as "01" or "breakfast_box", to test a single class.
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all", "checkpoint": default_checkpoint},
    # {"dataset": "btad", "path": data_root_btad, "class_name": "all", "checkpoint": default_checkpoint},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all", "checkpoint": default_checkpoint},
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all", "checkpoint": default_checkpoint},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all", "checkpoint": default_checkpoint},
    {"dataset": "hhled", "path": data_root_hhled, "class_name": "all", "checkpoint": default_checkpoint},
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

    if not ensure_meta_file(test_dataset, data_root):
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
