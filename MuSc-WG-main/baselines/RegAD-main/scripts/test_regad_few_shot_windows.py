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
    {"dataset": "visa", "path": data_root_visa, "class_name": "all"},
    # {"dataset": "btad", "path": data_root_btad, "class_name": "all"},
    # {"dataset": "mvtec_loco", "path": data_root_mvtec_loco, "class_name": "all"},
    # {"dataset": "microled", "path": data_root_microled, "class_name": "all"},
    # {"dataset": "miniled", "path": data_root_miniled, "class_name": "all"},
]

few_shots = [4]  # Define the number of few-shots to evaluate

# RegAD specific parameters
epochs = 50
batch_size = 32
lr = 0.0001
momentum = 0.9
inferences = 10
stn_mode = "rotation_scale"
seed = 10

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
            # Check if this is MVTec dataset, if so we can skip training and support set generation
            # since the pre-trained checkpoints and support sets are already provided.
            is_mvtec = "mvtec" in test_dataset.lower()
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            
            if not is_mvtec:
                # 1. Generate Support Set
                # RegAD requires pre-generating a support set file
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
                
                # 2. Train RegAD
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
                print(f"\n{'='*50}")
                print(f"Skipping Training & Support Set generation for MVTec class {c} (using provided checkpoints)...")
                print(f"{'='*50}\n")
            
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
