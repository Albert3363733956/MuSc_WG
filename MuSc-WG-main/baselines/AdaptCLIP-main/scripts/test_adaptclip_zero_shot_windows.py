import os
import subprocess
import sys
import importlib.util
from pathlib import Path

# Configuration
device = "0"
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def resolve_from_project(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_meta_file(dataset_name, data_root):
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        return True

    preprocessors = {
        "hhled": (project_root / "dataset" / "hhled.py", "HHLEDSolver"),
        "hhled_ad": (project_root / "dataset" / "hhled.py", "HHLEDSolver"),
    }
    if dataset_name not in preprocessors:
        print(f"Error: meta.json not found at {meta_path} for dataset {dataset_name}")
        return False

    module_path, class_name = preprocessors[dataset_name]
    spec = importlib.util.spec_from_file_location(f"{dataset_name}_solver", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver_cls = getattr(module, class_name)
    print(f"meta.json not found at {meta_path}; generating it for dataset {dataset_name}...")
    solver_cls(root=data_root).run()
    return meta_path.exists()

# Path to MVTec dataset
data_root_mvtec = r"C:\Users\Administrator\Desktop\dataset\MVTec"
data_root_mvtec_loco = r"C:\Users\Administrator\Desktop\dataset\MVTec_loco"
data_root_microled = r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD"
data_root_miniled = r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD"
data_root_hhled = r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD"

n_ctx = 12
vl_reduction = 4
pq_mid_dim = 128

# Using MVTec for both train and test as VisA is not available
# train_dataset = "mvtec"

shots = [0]

# Define test configurations
test_configs = [
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all", "train_dataset": "visa"}, # Test all MVTec LOCO classes
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all", "train_dataset": "visa"}, # Example
    {"dataset": "microled", "path": data_root_microled, "class_name": "all", "train_dataset": "visa"}, # Test all classes in microled
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all", "train_dataset": "visa"},   # Test all classes in miniled
    # {"dataset": "hhled", "path": data_root_hhled, "class_name": "all", "train_dataset": "visa"},       # Test all classes in hhled
]

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = resolve_from_project(config["path"])
    target_class = config["class_name"]
    train_dataset = config["train_dataset"]

    # Check if data root exists
    if not data_root.exists():
        print(f"Error: Data root not found at {data_root.resolve()} for dataset {test_dataset}")
        continue

    if not ensure_meta_file(test_dataset, data_root):
        continue

    for shot in shots:
        if shot == 0:
            seeds = [10]
        else:
            seeds = [10, 20, 30]
        
        for seed in seeds:
            base_dir = f"{n_ctx}_{vl_reduction}_{pq_mid_dim}_train_on_{train_dataset}_3adapters_batch8"
            
            # Paths
            save_dir = resolve_from_project(f"../../output/AdaptCLIP-main/{test_dataset}/zero_shot_{target_class}/{base_dir}_seed{seed}")
            model_dir = project_root / "adaptclip_checkpoints" / base_dir
            checkpoint_path = model_dir / "epoch_15.pth"
            
            # Check if checkpoint exists
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...")
                continue
                
            # Ensure save directory exists
            save_dir.mkdir(parents=True, exist_ok=True)

            # Construct the command
            cmd = [
                sys.executable, "test.py",
                "--dataset", test_dataset,
                "--test_data_path", str(data_root),
                "--seed", str(seed),
                "--k_shots", str(shot),
                "--checkpoint_path", str(checkpoint_path),
                "--save_path", str(save_dir),
                "--features_list", "6", "12", "18", "24",
                "--image_size", "518",
                "--batch_size", "8",
                "--n_ctx", str(n_ctx),
                "--vl_reduction", str(vl_reduction),
                "--pq_mid_dim", str(pq_mid_dim),
                "--visual_learner",
                "--textual_learner",
                "--pq_learner",
                "--pq_context",
                "--eval_metrics", "I-AUROC", "I-AP", "I-F1max", "P-AUROC", "P-AP", "P-F1max",
                "--visulize_bool",
                "--class_name", target_class
            ]

            # Set environment variable for CUDA
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            
            print(f"Running test for dataset={test_dataset}, class={target_class}, shot={shot}, seed={seed}...")
            
            try:
                subprocess.run(cmd, env=env, check=True, cwd=project_root)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
