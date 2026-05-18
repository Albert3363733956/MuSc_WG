import argparse
import os
import shutil
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.load_config import load_yaml

import warnings

warnings.filterwarnings("ignore")


def resolve_existing_path(path):
    if os.path.isabs(path):
        return path

    candidates = [
        os.path.abspath(path),
        os.path.abspath(os.path.join(REPO_ROOT, path)),
        os.path.abspath(os.path.join(SCRIPT_DIR, path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def resolve_output_dir(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def default_desktop_results_dir():
    return os.path.join(os.path.expanduser("~"), "Desktop", "MuSc_results")


def normalize_xlsx_filename(filename, default):
    filename = str(filename or default).strip()
    if not filename:
        filename = default
    filename = os.path.basename(filename)
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".xlsx":
        filename = f"{name}.xlsx" if ext else f"{filename}.xlsx"
    return filename


def get_args():
    parser = argparse.ArgumentParser(description="MuSc")
    parser.add_argument("--config", type=str, default="../configs/musc.yaml", help="config file path")
    parser.add_argument("--data_path", type=str, default=None, help="dataset path")
    parser.add_argument("--dataset_name", type=str, default=None, help="dataset name")
    parser.add_argument("--class_name", type=str, default=None, help="category")
    parser.add_argument("--device", type=int, default=None, help="gpu id")
    parser.add_argument("--output_dir", type=str, default=None, help="save results path")
    parser.add_argument("--vis", type=str, default=None, help="visualization")
    parser.add_argument("--vis_type", type=str, default=None, help="normalization type in visualization")
    parser.add_argument("--save_excel", type=str, default=None, help="save excel")
    parser.add_argument("--r_list", type=int, nargs="+", default=None, help="aggregation degrees of LNAMD")
    parser.add_argument("--feature_layers", type=int, nargs="+", default=None, help="feature layers")
    parser.add_argument("--backbone_name", type=str, default=None, help="backbone")
    parser.add_argument("--pretrained", type=str, default=None, help="pretrained datasets")
    parser.add_argument("--img_resize", type=int, default=None, help="image size")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--divide_num", type=int, default=None, help="the number of divided subsets")
    parser.add_argument("--module_profile", type=str, default="true", help="run module_profile.py after MuSc")
    parser.add_argument("--module_profile_output", type=str, default=None, help="module profile xlsx output path")
    parser.add_argument(
        "--module_profile_max_images",
        type=int,
        default=None,
        help="optional image limit per category for module profile",
    )
    parser.add_argument(
        "--desktop_results_dir",
        type=str,
        default=None,
        help="folder on Desktop for saving copies of results.xlsx and musc_module_profile.xlsx",
    )
    parser.add_argument("--desktop_results_filename", type=str, default=None, help="desktop MuSc xlsx filename")
    parser.add_argument("--module_profile_filename", type=str, default=None, help="desktop module profile xlsx filename")
    return parser.parse_args()


def load_args(cfg, args):
    if args.data_path is not None:
        cfg["datasets"]["data_path"] = args.data_path
    assert os.path.exists(cfg["datasets"]["data_path"]), f"The dataset path {cfg['datasets']['data_path']} does not exist."

    if args.dataset_name is not None:
        cfg["datasets"]["dataset_name"] = args.dataset_name
    if args.class_name is not None:
        cfg["datasets"]["class_name"] = args.class_name
    if args.device is not None:
        cfg["device"] = args.device
    if isinstance(cfg["device"], int):
        cfg["device"] = str(cfg["device"])
    if args.output_dir is not None:
        cfg["testing"]["output_dir"] = args.output_dir
    cfg["testing"]["output_dir"] = resolve_output_dir(cfg["testing"]["output_dir"])
    os.makedirs(cfg["testing"]["output_dir"], exist_ok=True)

    if args.vis is not None:
        cfg["testing"]["vis"] = args.vis.lower() == "true"
    if args.vis_type is not None:
        cfg["testing"]["vis_type"] = args.vis_type
    if args.save_excel is not None:
        cfg["testing"]["save_excel"] = args.save_excel.lower() == "true"
    if args.r_list is not None:
        cfg["models"]["r_list"] = args.r_list
    if isinstance(cfg["models"]["r_list"], int):
        cfg["models"]["r_list"] = [cfg["models"]["r_list"]]
    if args.feature_layers is not None:
        cfg["models"]["feature_layers"] = args.feature_layers
    if isinstance(cfg["models"]["feature_layers"], int):
        cfg["models"]["feature_layers"] = [cfg["models"]["feature_layers"]]
    if args.backbone_name is not None:
        cfg["models"]["backbone_name"] = args.backbone_name
    if args.pretrained is not None:
        cfg["models"]["pretrained"] = args.pretrained
    if args.img_resize is not None:
        cfg["datasets"]["img_resize"] = args.img_resize
    if args.batch_size is not None:
        cfg["models"]["batch_size"] = args.batch_size
    if args.divide_num is not None:
        cfg["datasets"]["divide_num"] = args.divide_num
    return cfg


def is_true(value):
    return str(value).lower() in ("1", "true", "yes", "y")


def add_optional_arg(cmd, name, value):
    if value is None:
        return
    if isinstance(value, list):
        if len(value) > 0:
            cmd.append(name)
            cmd.extend([str(v) for v in value])
    else:
        cmd.extend([name, str(value)])


def clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception as exc:
        print(f"Warning: failed to clear CUDA cache before module profile: {exc}")


def prepare_desktop_results(cfg, args):
    testing_cfg = cfg.get("testing", {})
    if args.desktop_results_dir is None:
        args.desktop_results_dir = testing_cfg.get("desktop_results_dir", default_desktop_results_dir())
    args.desktop_results_dir = os.path.abspath(os.path.expanduser(args.desktop_results_dir))
    os.makedirs(args.desktop_results_dir, exist_ok=True)

    if args.desktop_results_filename is None:
        args.desktop_results_filename = testing_cfg.get("desktop_results_filename", "results.xlsx")
    args.desktop_results_filename = normalize_xlsx_filename(args.desktop_results_filename, "results.xlsx")

    if args.module_profile_output is None:
        if args.module_profile_filename is None:
            args.module_profile_filename = testing_cfg.get("module_profile_filename", "musc_module_profile.xlsx")
        args.module_profile_filename = normalize_xlsx_filename(args.module_profile_filename, "musc_module_profile.xlsx")
        args.module_profile_output = os.path.join(args.desktop_results_dir, args.module_profile_filename)


def copy_excel_to_desktop(model, args):
    source = os.path.join(model.output_dir, "results.xlsx")
    target = os.path.join(args.desktop_results_dir, args.desktop_results_filename)
    if os.path.exists(source):
        shutil.copy2(source, target)
        print(f"MuSc Excel results copied to: {target}")
    else:
        print(f"Warning: Excel results were not found, so no xlsx file was copied: {source}")


def run_module_profile(args):
    script_path = os.path.join(REPO_ROOT, "models", "module_profile.py")
    cmd = [sys.executable, script_path]

    add_optional_arg(cmd, "--config", args.config)
    add_optional_arg(cmd, "--data_path", args.data_path)
    add_optional_arg(cmd, "--dataset_name", args.dataset_name)
    add_optional_arg(cmd, "--class_name", args.class_name)
    add_optional_arg(cmd, "--device", args.device)
    add_optional_arg(cmd, "--output_dir", args.output_dir)
    add_optional_arg(cmd, "--vis", args.vis)
    add_optional_arg(cmd, "--vis_type", args.vis_type)
    add_optional_arg(cmd, "--save_excel", args.save_excel)
    add_optional_arg(cmd, "--r_list", args.r_list)
    add_optional_arg(cmd, "--feature_layers", args.feature_layers)
    add_optional_arg(cmd, "--backbone_name", args.backbone_name)
    add_optional_arg(cmd, "--pretrained", args.pretrained)
    add_optional_arg(cmd, "--img_resize", args.img_resize)
    add_optional_arg(cmd, "--batch_size", args.batch_size)
    add_optional_arg(cmd, "--divide_num", args.divide_num)
    add_optional_arg(cmd, "--output_excel", args.module_profile_output)
    add_optional_arg(cmd, "--max_images", args.module_profile_max_images)

    print("Running MuSc module profile...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    args = get_args()
    args.config = resolve_existing_path(args.config)
    cfg = load_yaml(args.config)
    prepare_desktop_results(cfg, args)
    cfg = load_args(cfg, args)
    print(cfg)

    from models.musc import MuSc

    seed = 42
    model = MuSc(cfg, seed=seed)
    model.main()
    copy_excel_to_desktop(model, args)

    del model
    clear_cuda_cache()

    if is_true(args.module_profile):
        run_module_profile(args)
