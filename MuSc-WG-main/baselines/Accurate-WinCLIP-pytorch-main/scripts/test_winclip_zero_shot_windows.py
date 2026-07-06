import csv
import json
import os
import subprocess
import sys
from pathlib import Path

# Configuration
device = "0"
project_root = Path(__file__).resolve().parents[1]
repo_root = project_root.parents[1]
output_root = repo_root / "output" / "WinCLIP"

# Path to datasets
data_root_mvtec = r"C:\Users\Administrator\Desktop\dataset\MVTec"
data_root_visa = os.environ.get("VISA_DATA_ROOT", r"C:\Users\Administrator\Desktop\dataset\visa")
data_root_btad = os.environ.get("BTAD_DATA_ROOT", r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed")
data_root_mvtec_loco = r"C:\Users\Administrator\Desktop\dataset\MVTec_loco"
data_root_microled = r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD"
data_root_miniled = r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD"
data_root_hhled = r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD"

VISA_CLASS_NAMES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def ensure_visa_meta(data_root):
    data_root = Path(data_root)
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        return

    split_csv = data_root / "split_csv" / "1cls.csv"
    if not split_csv.exists():
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

    with split_csv.open("r", newline="", encoding="utf-8-sig") as f:
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

    with meta_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(info, indent=4) + "\n")
    print("normal_samples", normal_samples, "anomaly_samples", anomaly_samples)


def _relative_posix(path, root):
    return path.relative_to(root).as_posix()


def _image_files(path):
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_mask(mask_dir, image_path):
    masks = {p.stem: p for p in _image_files(mask_dir)}
    mask_path = masks.get(image_path.stem)
    if mask_path is None:
        raise FileNotFoundError(f"Missing BTAD mask for {image_path}: expected stem {image_path.stem} in {mask_dir}")
    return mask_path


def ensure_btad_meta(data_root):
    data_root = Path(data_root)
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        return

    class_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No BTAD class folders found under {data_root}")

    print(f"Generating BTAD meta.json under {data_root}...")
    info = {"train": {}, "test": {}}
    normal_samples = 0
    anomaly_samples = 0

    for class_dir in class_dirs:
        cls_name = class_dir.name
        for phase in ["train", "test"]:
            phase_dir = class_dir / phase
            samples = []
            specie_dirs = sorted(p for p in phase_dir.iterdir() if p.is_dir()) if phase_dir.exists() else []
            for specie_dir in specie_dirs:
                specie_name = specie_dir.name
                is_abnormal = specie_name != "ok"
                for img_path in _image_files(specie_dir):
                    mask_path = ""
                    if is_abnormal:
                        mask_path = _relative_posix(_find_mask(class_dir / "ground_truth" / specie_name, img_path), data_root)
                    samples.append(
                        {
                            "img_path": _relative_posix(img_path, data_root),
                            "mask_path": mask_path,
                            "cls_name": cls_name,
                            "specie_name": specie_name,
                            "anomaly": 1 if is_abnormal else 0,
                        }
                    )
                    if phase == "test":
                        if is_abnormal:
                            anomaly_samples += 1
                        else:
                            normal_samples += 1
            info[phase][cls_name] = samples

    with meta_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(info, indent=4) + "\n")
    print("normal_samples", normal_samples, "anomaly_samples", anomaly_samples)

# Define test configurations
# Uncomment the configuration you want to run
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all"},
    # {"dataset": "visa", "path": data_root_visa, "class_name": "fryum"},
    # {"dataset": "visa", "path": data_root_visa, "class_name": "pipe_fryum"},
    # {"dataset": "visa", "path": data_root_visa, "class_name": "pcb4"},
    {"dataset": "btad", "path": data_root_btad, "class_name": "all"},
    # {"dataset": "btad", "path": data_root_btad, "class_name": "01"},
    # {"dataset": "btad", "path": data_root_btad, "class_name": "02"},
    # {"dataset": "btad", "path": data_root_btad, "class_name": "03"},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all"},
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all"},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all"},
    # {"dataset": "hhled", "path": data_root_hhled, "class_name": "all"},
]

few_shot = 0
base_dir_prefix = "winclip_"
model_name = "ViT-B-16-plus-240"
pretrained_weights = "openai"
image_size = 240

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config.get("class_name", "all")

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue
    dataset_key = test_dataset.lower()
    if dataset_key == "visa":
        ensure_visa_meta(data_root)
    elif dataset_key == "btad":
        ensure_btad_meta(data_root)

    # Paths
    base_dir = f"{base_dir_prefix}{test_dataset}"
    if target_class.lower() == "all":
        save_dir = output_root / test_dataset / "zero_shot_all"
    else:
        save_dir = output_root / test_dataset / f"zero_shot_{target_class}"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the command
    cmd = [
        sys.executable, str(project_root / "reproduce_WinCLIP.py"),
        "--dataset", test_dataset,
        "--data_path", data_root,
        "--save_path", str(save_dir),
        "--model", model_name,
        "--pretrained", pretrained_weights,
        "--k_shot", str(few_shot),
        "--image_size", str(image_size),
        "--class_name", target_class,
        "--visulize_bool"
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running test for dataset={test_dataset}, class={target_class}...")
    
    try:
        subprocess.run(cmd, env=env, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
