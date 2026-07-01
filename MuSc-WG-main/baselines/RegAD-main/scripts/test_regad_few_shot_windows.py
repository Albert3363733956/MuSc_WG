import os
import subprocess
import sys
import json

# Configuration
device = "0"
# Path to datasets
data_root_mvtec = r"C:\Users\Administrator\Desktop\dataset\MVTec"
data_root_visa = r"C:\Users\Administrator\Desktop\dataset\visa"
data_root_btad = r"C:\Users\Administrator\Desktop\dataset\BTech_Dataset_transformed"
data_root_mvtec_loco = r"C:\Users\Administrator\Desktop\dataset\MVTec_loco"
data_root_microled = r"C:\Users\Administrator\Desktop\dataset\LED\microled_AD"
data_root_miniled = r"C:\Users\Administrator\Desktop\dataset\LED\miniled_AD"

# Output directory (Project root / output / RegAD)
output_dir = "../../output/RegAD"

# Define test configurations
# Uncomment the configuration you want to run
test_configs = [
    # {"dataset": "mvtec", "path": data_root_mvtec, "class_name": "all"},
    # {"dataset": "visa", "path": data_root_visa, "class_name": "all"},
    # {"dataset": "btad", "path": data_root_btad, "class_name": "all"},
    {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all"},
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all"},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all"},
]

few_shots = [4]  # Define the number of few-shots to evaluate

# RegAD specific parameters
epochs = 50
batch_size = 32
lr = 0.0001
momentum = 0.9
inferences = 1
stn_mode = "rotation_scale"
seed = 10

def checkpoint_paths(class_name, few_shot):
    return [
        os.path.join("save_checkpoints", str(few_shot), class_name, f"{class_name}_{few_shot}_rotation_scale_model.pt"),
        os.path.join("save_checkpoints", stn_mode, str(few_shot), class_name, f"{class_name}_{few_shot}_rotation_scale_model.pt"),
        os.path.join("logs_mvtec", stn_mode, str(few_shot), class_name, f"{class_name}_{few_shot}_{stn_mode}_model.pt"),
    ]

def has_checkpoint(class_name, few_shot):
    return any(os.path.exists(path) for path in checkpoint_paths(class_name, few_shot))

def support_set_path(class_name, few_shot):
    return os.path.join("support_set", class_name, f"{few_shot}_{inferences}.pt")

for config in test_configs:
    test_dataset = config["dataset"]
    data_root = config["path"]
    target_class = config["class_name"]

    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"Error: Data root not found at {os.path.abspath(data_root)} for dataset {test_dataset}")
        continue

    # Resolve target classes
    if target_class.lower() == "all":
        meta_path = os.path.join(data_root, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                classes = sorted(json.load(f)["test"].keys())
        else:
            classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    else:
        classes = [target_class]

    for c in classes:
        for few_shot in few_shots:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device

            checkpoint_ready = has_checkpoint(c, few_shot)
            support_ready = os.path.exists(support_set_path(c, few_shot))

            if checkpoint_ready and support_ready:
                print(f"\n{'='*50}")
                print(f"Skipping Training & Support Set generation for dataset={test_dataset}, class={c}, shot={few_shot} (existing checkpoint/support set found)...")
                print(f"{'='*50}\n")
            else:
                # 1. Generate Support Set
                # RegAD requires pre-generating a support set file
                if not support_ready:
                    print(f"\n{'='*50}")
                    print(f"Generating Support Set for dataset={test_dataset}, class={c}, shot={few_shot}...")
                    print(f"{'='*50}\n")

                    support_cmd = [
                        sys.executable, "generate_support_set.py",
                        "--obj", c,
                        "--data_path", data_root,
                        "--shot", str(few_shot),
                        "--inferences", str(inferences)
                    ]

                    try:
                        subprocess.run(support_cmd, env=env, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running support set generation for {c} ({few_shot}-shot): {e}")
                        continue
                else:
                    print(f"Using existing support set: {support_set_path(c, few_shot)}")

                # 2. Train RegAD
                if not checkpoint_ready:
                    print(f"\n{'='*50}")
                    print(f"Training RegAD for dataset={test_dataset}, class={c}, shot={few_shot}...")
                    print(f"{'='*50}\n")

                    train_cmd = [
                        sys.executable, "train.py",
                        "--obj", c,
                        "--data_type", test_dataset,
                        "--data_path", data_root,
                        "--shot", str(few_shot),
                        "--epochs", str(epochs),
                        "--batch_size", str(batch_size),
                        "--lr", str(lr),
                        "--momentum", str(momentum),
                        "--inferences", str(inferences),
                        "--stn_mode", stn_mode,
                        "--seed", str(seed)
                    ]

                    try:
                        subprocess.run(train_cmd, env=env, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running training for {c} ({few_shot}-shot): {e}")
                        continue
                else:
                    print(f"Using existing checkpoint for dataset={test_dataset}, class={c}, shot={few_shot}.")
            
            # 3. Test RegAD
            print(f"\n{'='*50}")
            print(f"Testing RegAD for dataset={test_dataset}, class={c}, shot={few_shot}...")
            print(f"{'='*50}\n")
            
            test_cmd = [
                sys.executable, "test.py",
                "--obj", c,
                "--data_type", test_dataset,
                "--data_path", data_root,
                "--shot", str(few_shot),
                "--inferences", str(inferences),
                "--stn_mode", stn_mode,
                "--seed", str(seed),
                "--output_dir", output_dir,
                "--visulize_bool"
            ]
            
            try:
                subprocess.run(test_cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running testing for {c} ({few_shot}-shot): {e}")
                continue
