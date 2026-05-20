import torch
from tqdm import tqdm


"""
MSM 2.0: speed-oriented MSM implementation.

This version keeps the original MSM scoring definition and the original
torch.cdist-based full-reference computation, but removes avoidable overhead:
- preallocates the output matrix instead of repeatedly torch.cat-ing rows;
- avoids the second topk by slicing the already selected nearest values;
- skips sorting when the result order is not needed;
- runs under inference_mode because MSM is used only for inference.
"""


def _resolve_topk(reference_num, topmin_min, topmin_max):
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(reference_num * k_max)
    if k_min < 1:
        k_min = int(reference_num * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    if k_max <= 0:
        raise ValueError(
            f"topmin_max={topmin_max} selects no reference images from {reference_num} candidates."
        )
    if k_min < 0 or k_min >= k_max:
        raise ValueError(f"Invalid topmin interval: k_min={k_min}, k_max={k_max}.")
    return int(k_min), int(k_max)


@torch.inference_mode()
def compute_scores_fast(
    Z,
    i,
    device,
    topmin_min=0,
    topmin_max=0.3,
    gamma=1.0,
    use_spot_weight=False,
):
    image_num, patch_num, c = Z.shape
    Z_ref = torch.cat((Z[:i], Z[i + 1:]), dim=0)
    patch2image = torch.cdist(Z[i:i + 1], Z_ref.reshape(-1, c)).reshape(
        patch_num,
        image_num - 1,
        patch_num,
    )
    patch2image = torch.min(patch2image, -1)[0]

    k_min, k_max = _resolve_topk(patch2image.shape[1], topmin_min, topmin_max)
    need_sorted = use_spot_weight or k_min > 0
    vals = torch.topk(
        patch2image.float(),
        k_max,
        largest=False,
        sorted=need_sorted,
    ).values

    if use_spot_weight:
        d_nearest = vals[:, 0] if need_sorted else vals.min(dim=1).values

    if k_min > 0:
        score = vals[:, k_min:k_max].mean(dim=1)
    else:
        score = vals.mean(dim=1)

    if use_spot_weight:
        score = torch.sqrt(score * d_nearest)

    if gamma != 1.0:
        score = torch.pow(score, gamma)

    return score


def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # Reference implementation kept for debugging small tensors.
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat(
                (
                    patch2image,
                    torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1),
                ),
                dim=1,
            )

    k_min, k_max = _resolve_topk(patch2image.shape[1], topmin_min, topmin_max)
    vals = torch.topk(patch2image.float(), k_max, largest=False, sorted=(k_min > 0)).values
    if k_min > 0:
        vals = vals[:, k_min:k_max]
    return torch.mean(vals, dim=1)


@torch.inference_mode()
def MSM(Z, device, topmin_min=0, topmin_max=0.3, gamma=1.0, use_spot_weight=False):
    image_num, patch_num, _ = Z.shape
    anomaly_scores_matrix = torch.empty(
        (image_num, patch_num),
        dtype=torch.double,
        device=device,
    )
    for i in tqdm(range(image_num)):
        anomaly_scores_matrix[i] = compute_scores_fast(
            Z,
            i,
            device,
            topmin_min,
            topmin_max,
            gamma,
            use_spot_weight,
        ).double()
    return anomaly_scores_matrix


if __name__ == "__main__":
    import time

    device = "cuda:0"
    Z = torch.rand(200, 1369, 1024).to(device)
    torch.cuda.synchronize()
    start_time = time.time()
    MSM(Z, device)
    torch.cuda.synchronize()
    end_time = time.time()
    print((end_time - start_time) * 1000)
