import csv
import json
import os
import subprocess
import sys

# Configuration
device = "0"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_from_project(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


# Path to datasets
data_root_mvtec = r"C:\Users\Administrator\Desktop\dataset\MVTec"
data_root_visa = os.environ.get("VISA_DATA_ROOT", r"C:\Users\Administrator\Desktop\dataset\visa")
data_root_mvtec_loco = r"C:\Users\Administrator\Desktop\dataset\MVTec_loco"
data_root_microled = r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD"
data_root_miniled = r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD"
data_root_hhled = r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD"

default_checkpoint = os.environ.get(
    "ANOMALYCLIP_CHECKPOINT",
    "./checkpoints/9_12_4_multiscale/epoch_15.pth",
)

VISA_CLASS_NAMES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def ensure_visa_meta(data_root):
    meta_path = os.path.join(data_root, "meta.json")
    if os.path.exists(meta_path):
        return

    split_csv = os.path.join(data_root, "split_csv", "1cls.csv")
    if not os.path.exists(split_csv):
        raise FileNotFoundError(
            f"ViSA split file not found: {split_csv}. "
            "Please make sure data_root_visa points to the prepared ViSA dataset root."
        )

    print(f"Generating ViSA meta.json from {split_csv}...")
    info = {
        "train": {cls_name: [] for cls_name in VISA_CLASS_NAMES},
        "test": {cls_name: [] for cls_name in VISA_CLASS_NAMES},
    }
    normal_samples = 0
    anomaly_samples = 0

    with open(split_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row["split"]
            cls_name = row["object"]
            is_abnormal = row["label"] == "anomaly"
            info[phase][cls_name].append(
                {
                    "img_path": row["image"],
                    "mask_path": row["mask"] if is_abnormal else "",
                    "cls_name": cls_name,
                    "specie_name": "",
                    "anomaly": 1 if is_abnormal else 0,
                }
            )
            if phase == "test":
                if is_abnormal:
                    anomaly_samples += 1
                else:
                    normal_samples += 1

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(info, indent=4) + "\n")
    print("normal_samples", normal_samples, "anomaly_samples", anomaly_samples)

# Define test configurations
# Uncomment the configuration you want to run
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "transistor", "checkpoint": "./checkpoints/9_12_4_multiscale/epoch_15.pth"}, 
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all", "checkpoint": default_checkpoint},
    {"dataset": "visa", "path": data_root_visa, "class_name": "fryum", "checkpoint": default_checkpoint},
    {"dataset": "visa", "path": data_root_visa, "class_name": "pipe_fryum", "checkpoint": default_checkpoint},
    {"dataset": "visa", "path": data_root_visa, "class_name": "pcb4", "checkpoint": default_checkpoint},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all", "checkpoint": "./checkpoints/9_12_4_multiscale/epoch_15.pth"},
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all", "checkpoint": "./checkpoints/9_12_4_multiscale/epoch_15.pth"},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all", "checkpoint": "./checkpoints/9_12_4_multiscale/epoch_15.pth"},
    # {"dataset": "hhled", "path": data_root_hhled, "class_name": "all", "checkpoint": "./checkpoints/9_12_4_multiscale/epoch_15.pth"},
]

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = resolve_from_project(config["path"])
    target_class = config["class_name"]
    checkpoint_path = resolve_from_project(config["checkpoint"])

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue
    if test_dataset.lower() == "visa":
        ensure_visa_meta(data_root)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {os.path.abspath(checkpoint_path)}")
        continue

    # Paths
    if target_class.lower() == "all":
        save_dir = resolve_from_project(f"../../output/AnomalyCLIP/{test_dataset}/zero_shot_all")
    else:
        save_dir = resolve_from_project(f"../../output/AnomalyCLIP/{test_dataset}/zero_shot_{target_class}")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the command
    cmd = [
        sys.executable, "test.py",
        "--dataset", test_dataset,
        "--data_path", data_root,
        "--checkpoint_path", checkpoint_path,
        "--save_path", save_dir,
        "--obj_name", target_class,
        "--features_list", "24",
        "--image_size", "518",
        "--depth", "9",
        "--n_ctx", "12",
        "--t_n_ctx", "4",
        "--visulize_bool", "True"
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running test for dataset={test_dataset}, class={target_class}...")
    
    try:
        subprocess.run(cmd, env=env, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
