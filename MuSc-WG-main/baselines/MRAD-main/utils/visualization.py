import cv2
import os
import hashlib
from utils.transforms import normalize
import numpy as np
import torch
from PIL import Image
# def visualizer(pathes, anomaly_map, img_size, save_path, cls_name):
#     for idx, path in enumerate(pathes):
#         cls = path.split('/')[-2]
#         filename = path.split('/')[-1]
#         vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
#         mask = normalize(anomaly_map[idx])
#         vis = apply_ad_scoremap(vis, mask)
#         vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
#         save_vis = os.path.join(save_path, 'imgs', cls_name[idx], cls)
#         if not os.path.exists(save_vis):
#             os.makedirs(save_vis)
#         cv2.imwrite(os.path.join(save_vis, filename), vis)

# def apply_ad_scoremap(image, scoremap, alpha=0.5):
#     np_image = np.asarray(image, dtype=float)
#     scoremap = (scoremap * 255).astype(np.uint8)
#     scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
#     scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
#     return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def make_output_base(rel_path, max_len=72):
    base = os.path.splitext(rel_path)[0]
    if len(base) <= max_len:
        return base

    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:10]
    parts = base.split("-")
    prefix = "-".join(parts[:3]) if len(parts) >= 3 else base
    prefix = prefix[:max_len - len(digest) - 1].rstrip("-_. ")
    return f"{prefix}-{digest}"

def read_image_rgb(img_path, img_size):
    return np.asarray(
        Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    ).copy()

def save_rgb_image(path, image):
    Image.fromarray(image.astype(np.uint8)).save(path)

# def visualizer(path, mask,anomaly_map, img_size):
#     filename = os.path.basename(path)
#     dirname = os.path.dirname(path)
#     vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
#     mask = normalize(anomaly_map[0])
#     vis = apply_ad_scoremap(vis, mask)
#     vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
#     save_vis = os.path.join(dirname, f'anomaly_map_{filename}')
#     print(save_vis)
#     cv2.imwrite(save_vis, vis)
def draw_mask_contour(image, mask):
    """
    image: numpy H×W×3 (RGB)
    mask: numpy H×W, binary 0/1
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = image.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)  # 绿色描边
    return vis
def visualizer(img_path, gt_mask, anomaly_map, save_dir, img_size=518, data_dir=None):
    """
    img_path: 原图路径 (str)
    gt_mask: torch.Tensor [1,1,H,W] 或 numpy
    anomaly_map: torch.Tensor [1,H,W] 或 numpy
    save_dir: 保存的文件夹路径
    img_size: 输出图像大小 (默认518)
    """

    os.makedirs(save_dir, exist_ok=True)

    # 解析文件名 (去掉前面的路径，只保留 datasets 后面的部分)
    # Use os.path.normpath and string replacement to handle different OS path separators
    norm_img_path = os.path.normpath(img_path)
    norm_data_dir = os.path.normpath(data_dir) if data_dir is not None else None
    
    # Extract relative path robustly
    if norm_data_dir is not None and norm_data_dir in norm_img_path:
        rel_path = norm_img_path.replace(norm_data_dir, "").lstrip(os.sep)
    else:
        # Fallback if split fails
        rel_path = os.path.basename(img_path)
        
    rel_path = rel_path.replace(os.sep, "-").replace("/", "-")     # e.g. "bottle-test-broken_small-000.png"
    base = make_output_base(rel_path)       # e.g. "bottle-test-broken_small-000"

    # 读取原图并resize (RGB)
    ori = read_image_rgb(img_path, img_size)

    # ---------- GT 可视化 ----------
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.squeeze().cpu().numpy()
    gt_mask = cv2.resize(gt_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    gt_vis = draw_mask_contour(ori, gt_mask)
    save_gt = os.path.join(save_dir, f"{base}_gt.png")
    save_rgb_image(save_gt, gt_vis)

    # ---------- 异常图可视化 ----------
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
    if anomaly_map.ndim == 3 and anomaly_map.shape[0] == 1:
        anomaly_map = anomaly_map[0]
    anomaly_map = cv2.resize(anomaly_map, (img_size, img_size))
    anomaly_map = normalize(anomaly_map)
    vis = apply_ad_scoremap(ori, anomaly_map)
    save_vis = os.path.join(save_dir, f"{base}_MRAD.png")
    save_rgb_image(save_vis, vis)

    print(f"Saved:\n {save_gt}\n {save_vis}")
