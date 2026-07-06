import os
import subprocess
import sys
import importlib.util
from types import SimpleNamespace
from pathlib import Path

# Configuration
device = os.environ.get("MRAD_CUDA_VISIBLE_DEVICES", "0")
project_root = Path(__file__).resolve().parents[1]
output_root = project_root.parents[1] / "output" / "MRAD"


def ensure_meta_file(dataset_name, data_root):
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        return True

    preprocessors = {
        "mvtec": (project_root / "generate_dataset_json" / "mvtec.py", "MVTecSolver"),
        "visa": (project_root / "generate_dataset_json" / "visa.py", "VisASolver"),
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


def env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def resolve_torch_device(torch_module):
    if torch_module.cuda.is_available() and device.strip() not in ("", "-1", "cpu"):
        return "cuda:0"
    return "cpu"


def get_cache_name(dataset_name):
    dataset_key = dataset_name.lower()
    if dataset_key in ("mvtec", "mvtec_loco"):
        return "visa"
    if dataset_key == "visa":
        return "mvtec"
    return "visa"


def build_cache_files(cache_name, cache_root):
    source_root = dataset_roots.get(cache_name)
    if source_root is None:
        print(f"Error: no source dataset configured to build cache for {cache_name}")
        return False
    if not source_root.exists():
        print(f"Error: cache source dataset not found at {source_root.resolve()}")
        return False
    if not ensure_meta_file(cache_name, source_root):
        return False

    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import torch
        import AnomalyCLIP_lib
        from mrad import build_cache_model, build_patch_cache_model
        from utils.dataset import Dataset
        from utils.transforms import get_transform
    except ModuleNotFoundError as exc:
        print(f"Error: cannot build MRAD cache because Python package is missing: {exc.name}")
        return False

    runtime_device = resolve_torch_device(torch)
    image_size = int(os.environ.get("MRAD_IMAGE_SIZE", "518"))
    batch_size = int(os.environ.get("MRAD_CACHE_BATCH_SIZE", "1"))
    cache_split = os.environ.get("MRAD_CACHE_SPLIT", "test")
    cache_obj_name = os.environ.get("MRAD_CACHE_OBJ_NAME", "all")

    print(f"Building MRAD cache '{cache_name}' from {source_root.resolve()} on {runtime_device}...")
    transform_args = SimpleNamespace(image_size=image_size)
    preprocess, target_transform = get_transform(transform_args)
    cache_data = Dataset(
        root=str(source_root),
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=cache_name,
        mode=cache_split,
        obj_name=cache_obj_name,
    )
    if len(cache_data) == 0:
        print(f"Error: no samples found for cache dataset={cache_name}, split={cache_split}, obj_name={cache_obj_name}")
        return False

    cache_loader = torch.utils.data.DataLoader(
        cache_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    anomalyclip_params = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": 9,
        "learnabel_text_embedding_length": 4,
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=runtime_device, design_details=anomalyclip_params)
    model.eval()
    model.to(runtime_device)
    model.visual.DAPM_replace(DPAM_layer=24)

    cache_root.mkdir(parents=True, exist_ok=True)
    image_cache_path = cache_root / f"cache_model_{cache_name}.pt"
    patch_cache_path = cache_root / f"cache_patch_model_{cache_name}.pt"
    if image_cache_path.exists():
        print(f"Image cache already exists: {image_cache_path.resolve()}")
    else:
        build_cache_model(
            load_cache=False,
            clip_model=model,
            train_loader_cache=cache_loader,
            device=runtime_device,
            dir=str(image_cache_path),
        )
    if patch_cache_path.exists():
        print(f"Patch cache already exists: {patch_cache_path.resolve()}")
    else:
        build_patch_cache_model(
            load_cache=False,
            clip_model=model,
            train_loader_cache=cache_loader,
            device=runtime_device,
            dir=str(patch_cache_path),
        )
    return True


def ensure_cache_files(dataset_name):
    cache_name = get_cache_name(dataset_name)
    cache_root = project_root / "cache"
    required_files = [
        cache_root / f"cache_model_{cache_name}.pt",
        cache_root / f"cache_patch_model_{cache_name}.pt",
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if not missing_files:
        return True

    print(f"MRAD cache files for dataset {dataset_name} not found:")
    for path in missing_files:
        print(f"  - {path.resolve()}")

    if env_flag("MRAD_AUTO_BUILD_CACHE", True):
        if build_cache_files(cache_name, cache_root):
            missing_files = [path for path in required_files if not path.exists()]
            if not missing_files:
                return True
            print("Error: cache build finished, but required files are still missing:")
            for path in missing_files:
                print(f"  - {path.resolve()}")
            return False

    print("Set MRAD_AUTO_BUILD_CACHE=1 to build them locally, or download the official MRAD memory banks into ./cache.")
    return False

# Path to datasets
data_root_mvtec = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec")
data_root_mvtec_loco = Path(r"C:\Users\Administrator\Desktop\dataset\MVTec_loco")
data_root_visa = Path(r"C:\Users\Administrator\Desktop\dataset\visa")
data_root_microled = Path(r"C:\Users\Administrator\Desktop\dataset\LED2\microled_AD")
data_root_miniled = Path(r"C:\Users\Administrator\Desktop\dataset\LED2\miniled_AD")
data_root_hhled = Path(r"C:\Users\Administrator\Desktop\dataset\LED2\hhled_AD")

default_checkpoint = Path(os.environ.get("MRAD_CHECKPOINT", project_root / "checkpoints" / "test_on_mvtec.pth"))

# Define test configurations.
dataset_roots = {
    "mvtec": data_root_mvtec,
    "mvtec_loco": data_root_mvtec_loco,
    "visa": data_root_visa,
    "microled": data_root_microled,
    "miniled": data_root_miniled,
    "hhled": data_root_hhled,
}
default_target_classes = {
    "mvtec": ["all"],
    "mvtec_loco": ["all"],
    "visa": ["all"],
    "microled": ["all"],
    "miniled": ["all"],
    "hhled": ["all"],
}

selected_dataset = os.environ.get("MRAD_TEST_DATASET", "visa").lower()
target_classes_env = os.environ.get("MRAD_TARGET_CLASSES")
target_classes = (
    [name.strip() for name in target_classes_env.split(",") if name.strip()]
    if target_classes_env
    else default_target_classes.get(selected_dataset, ["all"])
)
if selected_dataset not in dataset_roots:
    raise ValueError(
        f"Unsupported MRAD_TEST_DATASET={selected_dataset!r}. "
        f"Available values: {', '.join(sorted(dataset_roots))}"
    )

test_configs = [
    {"dataset": selected_dataset, "path": dataset_roots[selected_dataset], "class_name": class_name, "checkpoint": default_checkpoint}
    for class_name in target_classes
]

cache_check_results = {}
prepare_cache_only = env_flag("MRAD_PREPARE_CACHE_ONLY", False)

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
    if test_dataset not in cache_check_results:
        cache_check_results[test_dataset] = ensure_cache_files(test_dataset)
    if not cache_check_results[test_dataset]:
        continue
    if prepare_cache_only:
        print(f"MRAD cache is ready for dataset={test_dataset}.")
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
        "--visulize_bool", os.environ.get("MRAD_VISUALIZE", "True"),
        "--compute_pixel_aupro", os.environ.get("MRAD_COMPUTE_PIXEL_AUPRO", "False"),
    ]
    
    # Set environment variable for CUDA
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    
    print(f"Running test for dataset={test_dataset}, class={target_class}...")
    
    try:
        subprocess.run(cmd, env=env, check=True, cwd=project_root)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
