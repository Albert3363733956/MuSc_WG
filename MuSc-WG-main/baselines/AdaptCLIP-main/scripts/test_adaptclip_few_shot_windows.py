import os
import importlib.util
from pathlib import Path
import subprocess
import sys


# Configuration
device = "0"
project_root = Path(__file__).resolve().parents[1]
output_root = (project_root / ".." / ".." / "output" / "AdaptCLIP-main").resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


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

# Paths to datasets
data_root_mvtec = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec")
data_root_btad = Path(r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed")
data_root_mvtec_loco = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec_loco")
data_root_microled = Path(r"C:\Users\Administrator\Desktop\dataset\LED\microled_AD")
data_root_miniled = Path(r"C:\Users\Administrator\Desktop\dataset\LED\miniled_AD")
data_root_hhled = Path(r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD")

n_ctx = 12
vl_reduction = 4
pq_mid_dim = 128
train_dataset = "visa"

shots = [4]
seeds = [10]

# Set "class_name" to a concrete class, such as "01" or "breakfast_box", to test a single class.
test_configs = [
    # {"dataset": "btad", "path": data_root_btad, "class_name": "all", "train_dataset": train_dataset},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all", "train_dataset": train_dataset},
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "transistor", "train_dataset": train_dataset},
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all", "train_dataset": train_dataset},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all", "train_dataset": train_dataset},
    {"dataset": "hhled", "path": data_root_hhled, "class_name": "all", "train_dataset": train_dataset},
]


for config in test_configs:
    test_dataset = config["dataset"]
    data_root = Path(config["path"])
    target_class = config.get("class_name", "all")
    train_dataset_name = config.get("train_dataset", train_dataset)

    if not data_root.exists():
        print(f"Error: Data root not found at {data_root.resolve()} for dataset {test_dataset}")
        continue

    if not ensure_meta_file(test_dataset, data_root):
        continue

    for shot in shots:
        for seed in seeds:
            base_dir = f"{n_ctx}_{vl_reduction}_{pq_mid_dim}_train_on_{train_dataset_name}_3adapters_batch8"

            save_dir = output_root / test_dataset / f"few_shot_{shot}_{target_class}" / f"{base_dir}_seed{seed}"
            checkpoint_path = project_root / "adaptclip_checkpoints" / base_dir / "epoch_15.pth"

            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found at {checkpoint_path}, skipping...")
                continue

            save_dir.mkdir(parents=True, exist_ok=True)

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
                "--class_name", target_class,
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device

            print(f"\n{'=' * 50}")
            print(
                f"Running AdaptCLIP few-shot for dataset={test_dataset}, "
                f"class={target_class}, shot={shot}, seed={seed}..."
            )
            print(f"{'=' * 50}\n")

            try:
                subprocess.run(cmd, env=env, cwd=project_root, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command for {test_dataset} ({target_class}, {shot}-shot): {e}")
