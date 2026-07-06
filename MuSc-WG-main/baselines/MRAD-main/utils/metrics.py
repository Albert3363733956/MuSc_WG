from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
from scipy import ndimage

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    masks = masks.astype(bool)
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    if max_th == min_th:
        return 0.0
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []

    region_infos = []
    for mask in masks:
        labeled_mask, region_count = ndimage.label(mask)
        if region_count == 0:
            region_infos.append(None)
            continue
        region_ids = np.arange(1, region_count + 1)
        areas = ndimage.sum(mask, labeled_mask, index=region_ids)
        valid = areas > 0
        region_infos.append((labeled_mask, region_ids[valid], areas[valid]))

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, image_regions in zip(binary_amaps, region_infos):
            if image_regions is None:
                continue
            labeled_mask, region_ids, areas = image_regions
            tp_pixels = ndimage.sum(binary_amap, labeled_mask, index=region_ids)
            pro.extend(tp_pixels / areas)
        inverse_masks = np.logical_not(masks)
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        inverse_area = inverse_masks.sum()
        fpr = fp_pixels / inverse_area if inverse_area > 0 else 0.0
        pros.append(np.array(pro).mean() if pro else 0.0)
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if fprs.size < 2 or fprs.max() == fprs.min():
        return 0.0
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)
    elif metric == 'image-f1':
        precision, recall, _ = precision_recall_curve(gt, pr)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        performance = np.max(f1[np.isfinite(f1)])
    return performance
    # table.append(str(np.round(performance * 100, decimals=1)))


def pixel_level_metrics(results, obj, metric):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'pixel-auroc':
        performance = roc_auc_score(gt.ravel(), pr.ravel())
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr)
    elif metric == 'pixel-ap':
        performance = average_precision_score(gt.ravel(), pr.ravel())
    elif metric == 'pixel-f1':
        precisions, recalls, thresholds = precision_recall_curve(gt.ravel(), pr.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls+ 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        performance = np.max(f1_scores[np.isfinite(f1_scores)])
    return performance
    
