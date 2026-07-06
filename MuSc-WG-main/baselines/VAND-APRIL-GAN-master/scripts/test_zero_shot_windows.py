import json
import os
import subprocess
import sys
from pathlib import Path

# Configuration
device = "0"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_from_project(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(project_root, path))


# Path to datasets
data_root_mvtec = r"C:\Users\Administrator\Desktop\dataset\MVTec"
data_root_btad = os.environ.get("BTAD_DATA_ROOT", r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed")
data_root_mvtec_loco = r"C:\Users\Administrator\Desktop\dataset\MVTec_loco"
data_root_microled = r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD"
data_root_miniled = r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD"
data_root_hhled = r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD"

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def relative_posix(path, root):
    return path.relative_to(root).as_posix()


def image_files(path):
    if not path.exists():
        return []
    return sorted(
        file_path for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_mask(mask_dir, image_path):
    masks = {mask_path.stem: mask_path for mask_path in image_files(mask_dir)}
    mask_path = masks.get(image_path.stem)
    if mask_path is None:
        raise FileNotFoundError(
            f"Missing BTAD mask for {image_path}: expected stem {image_path.stem} in {mask_dir}"
        )
    return mask_path


def ensure_btad_meta(data_root):
    data_root = Path(data_root)
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        return

    class_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
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
            specie_dirs = sorted(path for path in phase_dir.iterdir() if path.is_dir()) if phase_dir.exists() else []
            for specie_dir in specie_dirs:
                specie_name = specie_dir.name
                is_abnormal = specie_name != "ok"
                for img_path in image_files(specie_dir):
                    mask_path = ""
                    if is_abnormal:
                        mask_path = relative_posix(
                            find_mask(class_dir / "ground_truth" / specie_name, img_path),
                            data_root,
                        )
                    samples.append(
                        {
                            "img_path": relative_posix(img_path, data_root),
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
    # {"dataset": "mvtec", "path": data_root_mvtec, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    {"dataset": "btad", "path": data_root_btad, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "btad", "path": data_root_btad, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "01"},
    # {"dataset": "btad", "path": data_root_btad, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "02"},
    # {"dataset": "btad", "path": data_root_btad, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "03"},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "microled", "path": data_root_microled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "miniled", "path": data_root_miniled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "hhled", "path": data_root_hhled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
]

# Base arguments
config_path = "./open_clip/model_configs/ViT-L-14-336.json"
model_name = "ViT-L-14-336"
features_list = ["6", "12", "18", "24"]
pretrained = "openai"
image_size = "518"
mode = "zero_shot"

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = resolve_from_project(config["path"])
    checkpoint_path = config["checkpoint"]
    class_name = config.get("class_name", "all")

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue
    if test_dataset.lower() == "btad":
        ensure_btad_meta(data_root)

    # Paths
    save_name = "zero_shot" if class_name.lower() == "all" else f"zero_shot_{class_name}"
    save_dir = resolve_from_project(f"../../output/VAND-APRIL-GAN-master/{test_dataset}/{save_name}")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct the main test command
    cmd = [
        sys.executable, "test.py",
        "--mode", mode,
        "--dataset", test_dataset,
        "--data_path", data_root,
        "--save_path", save_dir,
        "--config_path", config_path,
        "--checkpoint_path", checkpoint_path,
        "--model", model_name,
        "--features_list", *features_list,
        "--pretrained", pretrained,
        "--image_size", image_size,
        "--class_name", class_name,
        "--visulize_bool"
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running zero-shot test for dataset={test_dataset}...")
    
    try:
        subprocess.run(cmd, env=env, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
