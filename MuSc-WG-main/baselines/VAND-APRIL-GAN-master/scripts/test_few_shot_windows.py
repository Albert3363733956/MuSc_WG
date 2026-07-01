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
data_root_btad = r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed"
data_root_mvtec_loco = r"C:\Users\Administrator\Desktop\dataset\MVTec_loco"
data_root_microled = r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD"
data_root_miniled = r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD"
data_root_hhled = r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD"

# Define test configurations
# Uncomment the configuration you want to run
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "btad", "path": data_root_btad, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "microled", "path": data_root_microled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    # {"dataset": "miniled", "path": data_root_miniled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
    {"dataset": "hhled", "path": data_root_hhled, "checkpoint": "./exps/pretrained/visa_pretrained.pth", "class_name": "all"},
]

few_shots = [4]  # Define the number of few-shots to evaluate

# Base arguments
config_path = "./open_clip/model_configs/ViT-L-14-336.json"
model_name = "ViT-L-14-336"
features_list = ["6", "12", "18", "24"]
few_shot_features = ["6", "12", "18", "24"]
pretrained = "openai"
image_size = "518"
mode = "few_shot"

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = resolve_from_project(config["path"])
    checkpoint_path = config["checkpoint"]
    class_name = config.get("class_name", "all")

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    for few_shot in few_shots:
        # Paths
        save_dir = resolve_from_project(f"../../output/VAND-APRIL-GAN-master/{test_dataset}/few_shot_{few_shot}")
        
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
            "--few_shot_features", *few_shot_features,
            "--pretrained", pretrained,
            "--image_size", image_size,
            "--class_name", class_name,
            "--k_shot", str(few_shot),
            "--visulize_bool"
        ]
        
        # Set environment variable for CUDA
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        
        print(f"Running few-shot ({few_shot}-shot) test for dataset={test_dataset}...")
        
        try:
            subprocess.run(cmd, env=env, cwd=project_root, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
