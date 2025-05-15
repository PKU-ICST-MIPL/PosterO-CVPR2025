import torch
from torchvision.transforms.functional import to_tensor
import os
from PIL import Image
import numpy as np
import pickle as pkl
from pandas import read_csv

import multiprocessing
from functools import partial
from einops import rearrange, reduce, repeat
import itertools
import cv2

# format uniformation

def nameToImageTensor(base_dir, name):
    try:
        image = Image.open(os.path.join(base_dir, "input", f"{name}.png")).convert("RGB")
    except:
        image = Image.open(os.path.join(base_dir, "input", f"{name}.jpg")).convert("RGB")
    image = image.resize((240, 350))
    return to_tensor(image)

def nameToSaliencyTensor(base_dir, name):
    saliency = Image.open(os.path.join(base_dir, "saliency", f"{name}.png")).convert("L")
    saliency_sub = Image.open(os.path.join(base_dir, "saliency_sub", f"{name}.png")).convert("L")
    saliency = Image.fromarray(np.maximum(np.array(saliency), np.array(saliency_sub)))
    saliency = saliency.resize((240, 350))
    return to_tensor(saliency)

def nameToDensityTensor(base_dir, dataset, name):
    density = Image.open(os.path.join(base_dir, f"{dataset}_{name}.png")).convert("L")
    density = density.resize((240, 350))
    return to_tensor(density)

def ptsToBatch(pts):
    # 'center_x', 'center_y', 'width', 'height', 'label', 'id', 'image', 'saliency', 'mask']
    # [max_len, 10], [max_len, 10], [max_len, 10], [max_len, 10], [max_len, 10], List(max_len), [max_len, 3, 350, 240], [max_len, 1, 350, 240], [max_len, 10]
    center_xs = []
    center_ys = []
    widths = []
    heights = []
    labels = []
    ids = []
    masks = []
    
    for pt in pts:
        valid_n = len(pt["label"])
        max_len = 10
        center_x = torch.zeros(max_len)
        center_y = torch.zeros(max_len)
        width = torch.zeros(max_len)
        height = torch.zeros(max_len)
        label = torch.zeros(max_len)
        mask = torch.concat([torch.ones(min(max_len, valid_n)), torch.zeros(max(0, max_len - valid_n))]).bool()
        
        center_x[:valid_n] = torch.tensor(pt["center_x"][:max_len])
        center_y[:valid_n] = torch.tensor(pt["center_y"][:max_len])
        width[:valid_n] = torch.tensor(pt["width"][:max_len])
        height[:valid_n] = torch.tensor(pt["height"][:max_len])
        label[:valid_n] = torch.tensor(pt["label"][:max_len])
        pid = pt["id"]
        
        center_xs.append(center_x)
        center_ys.append(center_y)
        widths.append(width)
        heights.append(height)
        labels.append(label)
        ids.append(pid)
        masks.append(mask)

        
    center_xs = torch.stack(center_xs, dim=0)
    center_ys = torch.stack(center_ys, dim=0)
    widths = torch.stack(widths, dim=0)
    heights = torch.stack(heights, dim=0)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)

    batch = {
        "center_x": center_xs, 
        "center_y": center_ys, 
        "width": widths, 
        "height": heights, 
        "label": labels, 
        "id": ids, 
        "mask": masks
    }

    # print('center_x:', center_xs.shape)
    # print('center_y:', center_ys.shape)
    # print('width:', widths.shape)
    # print('height:', heights.shape)
    # print('label:', labels.shape)
    # print('id:', len(ids))
    # print('mask:', masks.shape)

    return batch

def gather_pt(file):
    data = []
    for pt_path in file:
        if os.path.exists(pt_path):
            pt = torch.load(pt_path, weights_only=False)
            pt = pt[0] # select highest ranker
        else:
            pt = (torch.zeros(1), torch.zeros(1, 4))

        d = {
            "center_x": pt[1][:, 0] + 0.5 * pt[1][:, 2],
            "center_y": pt[1][:, 1] + 0.5 * pt[1][:, 3],
            "width": pt[1][:, 2],
            "height": pt[1][:, 3],
            "id": os.path.splitext(os.path.split(pt_path)[-1])[0],
            "label": pt[0]
        }
        data.append(d)
    return data

def gather_pSplit(split, label_id):
    data = []
    failed = []
    for i in range(len(split)):
        generated = split[i]['generated']['layout'][0]
        if generated['cls_elem'] == []:
            generated['cls_elem'] = [0.0]
            generated['box_elem'] = [[0.0, 0.0, 0.0, 0.0]]
            failed.append(i)
            # continue
        
        # label to id
        if generated['cls_elem'][0] == 'canvas':
            generated['cls_elem'].pop(0)
            generated['box_elem'].pop(0)

        rm = []
        for j in range(len(generated['cls_elem'])):
            if generated['cls_elem'][j] in label_id:
                generated['cls_elem'][j] = label_id[generated['cls_elem'][j]]
            else:
                rm.append(j)

        if rm:
            for rmi in range(len(rm)):
                generated['cls_elem'].pop(rm[rmi] - rmi)
                generated['box_elem'].pop(rm[rmi] - rmi)

        generated['cls_elem'] = torch.tensor(generated['cls_elem'], dtype=float)
        generated['box_elem'] = torch.tensor(generated['box_elem'], dtype=float)
        if len(generated['cls_elem']) == 0:
            generated['cls_elem'] = torch.tensor([0.0])
            generated['box_elem'] = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
            
        generated['box_elem'][:, ::2] = torch.div(generated['box_elem'][:, ::2], 513.0)
        generated['box_elem'][:, 1::2] = torch.div(generated['box_elem'][:, 1::2], 750.0)
        d = {
            "center_x": generated['box_elem'][:, 0] + 0.5 * generated['box_elem'][:, 2],
            "center_y": generated['box_elem'][:, 1] + 0.5 * generated['box_elem'][:, 3],
            "width": generated['box_elem'][:, 2],
            "height": generated['box_elem'][:, 3],
            "id": os.path.splitext(os.path.split(split[i]['poster_path'])[-1])[0],
            "label": generated['cls_elem']
        }
        data.append(d)
    print("failed:", failed)
    return data

# utils

def convert_xywh_to_ltrb(bbox):
    assert len(bbox) == 4
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def is_list_of_dict(x):
    if isinstance(x, list):
        return all(isinstance(d, dict) for d in x)
    else:
        return False

def list_of_dict_to_dict_of_list(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}

# metric functions

def compute_validity(data, thresh=1e-3):
    """
    Ratio of valid elements to all elements in the layout used in PosterLayout,
    where the area must be greater than 0.1% of the canvas.
    For validity, higher values are better (in 0.0 - 1.0 range).
    """
    filtered_data = []
    N_numerator, N_denominator = 0, 0
    for d in data:
        is_valid = [(w * h > thresh) for (w, h) in zip(d["width"], d["height"])]
        N_denominator += len(is_valid)
        N_numerator += is_valid.count(True)

        filtered_d = {}
        for key, value in d.items():
            if isinstance(value, list):
                filtered_d[key] = []
                assert len(value) == len(
                    is_valid
                ), f"{len(value)} != {len(is_valid)}, value: {value}, is_valid: {is_valid}"
                for j in range(len(is_valid)):
                    if is_valid[j]:
                        filtered_d[key].append(value[j])
            else:
                filtered_d[key] = value
        filtered_data.append(filtered_d)

    validity = N_numerator / N_denominator
    return filtered_data, validity

def compute_overlay(batch, underlay_id):
    """
    See __compute_overlay for detailed description.
    """
    layouts = []
    for i in range(batch["label"].size(0)):
        new_mask = batch["mask"][i] & (
            batch["label"][i] != underlay_id
        )  # ignore underlay
        label = batch["label"][i][new_mask]
        bbox = []
        for key in ["center_x", "center_y", "width", "height"]:
            bbox.append(batch[key][i][new_mask])
        bbox = torch.stack(bbox, dim=-1)  # type: ignore
        layouts.append((np.array(bbox), np.array(label)))

    results: dict[str, list[float]] = {
        "overlay": run_parallel(__compute_overlay, layouts)
    }
    return results

def compute_alignment(batch):
    """
    Computes some alignment metrics that are different to each other in previous works.
    Lower values are generally better.
    Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    """
    xl, xc, xr, yt, yc, yb = _get_coords(batch)
    mask = batch["mask"]
    _, S = mask.size()

    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log10(1 - X)

    # original
    # return X.sum(-1) / mask.float().sum(-1)

    score = reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    Y = torch.stack([xl, xc, xr], dim=1)
    Y = rearrange(Y, "b x s -> b x 1 s") - rearrange(Y, "b x s -> b x s 1")

    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=Y.device)
    batch_mask[:, idx, idx] = True
    batch_mask = repeat(batch_mask, "b s1 s2 -> b x s1 s2", x=3)
    Y[batch_mask] = 1.0

    # Y = rearrange(Y.abs(), "b x s1 s2 -> b s1 x s2")
    # Y = reduce(Y, "b x s1 s2 -> b x", "min")
    # Y = rearrange(Y.abs(), " -> b s1 x s2")
    Y = reduce(Y.abs(), "b x s1 s2 -> b s1", "min")
    Y[Y == 1.0] = 0.0
    score_Y = reduce(Y, "b s -> b", "sum")

    results = {
        # "alignment-ACLayoutGAN": score,  # Because it may be confusing.
        "alignment-LayoutGAN++": score_normalized,
        # "alignment-NDN": score_Y,  # Because it may be confusing.
    }
    return {k: v.tolist() for (k, v) in results.items()}

def compute_underlay_effectiveness(batch, underlay_id):
    """
    See __compute_underlay_effectiveness for detailed description.
    """
    layouts = []
    for i in range(batch["label"].size(0)):
        mask = batch["mask"][i]
        label = batch["label"][i][mask]
        bbox = []
        for key in ["center_x", "center_y", "width", "height"]:
            bbox.append(batch[key][i][mask])
        bbox = torch.stack(bbox, dim=-1)  # type: ignore
        layouts.append((np.array(bbox), np.array(label)))

    results: dict[str, list[float]] = run_parallel(
        partial(__compute_underlay_effectiveness, underlay_id=underlay_id), layouts
    )
    return results

def compute_density_aware_metrics(batchs, base_dir, dataset):
    """
    - intention_coverage:
        Utilization rate of space suitable for arranging elements using density map,
        Higher values are generally better (in 0.0 - 1.0 range).
    - intention_conflict:
        Conflict rate of space suitable for arranging elements using density map,
        Lower values are generally better.
    """
    results = {"intention_coverage": [], "intention_conflict": []}
    for b_i in range(0, len(batchs["id"]), 20):
        batch = {k: v[b_i:b_i+20] for k, v in batchs.items()}
        
        batch["density"] = []
        for pid in batch["id"]:
            density = nameToDensityTensor(base_dir['density'], dataset, pid)
            batch["density"].append(density)
        
        batch["density"] = torch.stack(batch["density"], dim=0)
        
        B, _, H, W = batch["density"].size()
        xl, _, xr, yt, _, yb = _get_coords(batch)
        density = rearrange(batch["density"], "b 1 h w -> b h w")
        inv_density = 1.0 - density
    
        
        for i in range(B):
            mask = batch["mask"][i]
            left = (xl[i][mask] * W).round().int().tolist()
            top = (yt[i][mask] * H).round().int().tolist()
            right = (xr[i][mask] * W).round().int().tolist()
            bottom = (yb[i][mask] * H).round().int().tolist()
    
            bbox_mask = torch.zeros((H, W))
            for l, t, r, b in zip(left, top, right, bottom):
                bbox_mask[t:b, l:r] = 1
    
            # intention_coverage
            numerator = torch.sum(density[i] * bbox_mask)
            denominator = torch.sum(density[i])
            assert denominator > 0.0
            results["intention_coverage"].append((numerator / denominator).item())
            
            # intention_conflict
            numerator = torch.sum(inv_density[i] * bbox_mask)
            denominator = torch.sum(inv_density[i])
            assert denominator > 0.0
            results["intention_conflict"].append((numerator / denominator).item())

    return results
    

def compute_saliency_aware_metrics(batchs, base_dir, text_id, underlay_id):
    """
    - utilization:
        Utilization rate of space suitable for arranging elements,
        Higher values are generally better (in 0.0 - 1.0 range).
    - occlusion:
        Average saliency of areas covered by elements.
        Lower values are generally better (in 0.0 - 1.0 range).
    - unreadability:
        Non-flatness of regions that text elements are solely put on
        Lower values are generally better.
    """
    results = {"utilization": [], "occlusion": [], "unreadability": []}
    for b_i in range(0, len(batchs["id"]), 20):
        batch = {k: v[b_i:b_i+20] for k, v in batchs.items()}
        
        batch["image"] = []
        batch["saliency"] = []
        batch["density"] = []
        for pid in batch["id"]:
            image = nameToImageTensor(base_dir['general'], pid)
            saliency = nameToSaliencyTensor(base_dir['general'], pid)
            batch["image"].append(image)
            batch["saliency"].append(saliency)
        
        batch["image"] = torch.stack(batch["image"], dim=0)
        batch["saliency"] = torch.stack(batch["saliency"], dim=0)
        
        B, _, H, W = batch["saliency"].size()
        saliency = rearrange(batch["saliency"], "b 1 h w -> b h w")
        inv_saliency = 1.0 - saliency
        xl, _, xr, yt, _, yb = _get_coords(batch)
    
        
        for i in range(B):
            mask = batch["mask"][i]
            left = (xl[i][mask] * W).round().int().tolist()
            top = (yt[i][mask] * H).round().int().tolist()
            right = (xr[i][mask] * W).round().int().tolist()
            bottom = (yb[i][mask] * H).round().int().tolist()
    
            bbox_mask = torch.zeros((H, W))
            for l, t, r, b in zip(left, top, right, bottom):
                bbox_mask[t:b, l:r] = 1
    
            # utilization
            numerator = torch.sum(inv_saliency[i] * bbox_mask)
            denominator = torch.sum(inv_saliency[i])
            assert denominator > 0.0
            results["utilization"].append((numerator / denominator).item())
    
            # occlusion
            occlusion = saliency[i][bbox_mask.bool()]
            if len(occlusion) == 0:
                results["occlusion"].append(0.0)
            else:
                results["occlusion"].append(occlusion.mean().item())
    
            # unreadability
            # note: values are much smaller than repoted probably because
            # they compute gradient in 750*513
            bbox_mask_special = torch.zeros((H, W))
            label = batch["label"][i].tolist()
    
            for id_, l, t, r, b in zip(label, left, top, right, bottom):
                # get text area
                if id_ == text_id:
                    bbox_mask_special[t:b, l:r] = 1
            for id_, l, t, r, b in zip(label, left, top, right, bottom):
                # subtract underlay area
                if id_ == underlay_id:
                    bbox_mask_special[t:b, l:r] = 0
    
            g_xy = _extract_grad(batch["image"][i])
            unreadability = g_xy[bbox_mask_special.bool()]
            if len(unreadability) == 0:
                results["unreadability"].append(0.0)
            else:
                results["unreadability"].append(unreadability.mean().item())

    return results

# metric sub functions

def run_parallel(func, layouts, is_debug=True, n_jobs=None):
    """
    Assumption:
    each func returns a single value or dict where each element is a single value
    """
    if is_debug:
        scores = [func(layout) for layout in layouts]
    else:
        with multiprocessing.Pool(n_jobs) as p:
            scores = p.map(func, layouts)

    if is_list_of_dict(scores):
        filtered_scores = list_of_dict_to_dict_of_list(scores)
        for k in filtered_scores:
            filtered_scores[k] = [s for s in filtered_scores[k]] # if s is not None]
        return filtered_scores
    else:
        return [s for s in scores] # if s is not None]

def __compute_overlay(layout):
    """
    Average IoU except underlay components used in PosterLayout.
    Lower values are better (in 0.0 - 1.0 range).
    """
    bbox, _ = layout
    N = bbox.shape[0]
    if N in [0, 1]:
        return 0 # None # no overlap in principle

    ii, jj = _list_all_pair_indices(bbox)
    iou = iou_func_factory("iou")(bbox[ii], bbox[jj])
    result = iou.mean().item()
    return result

def _list_all_pair_indices(bbox):
    """
    Generate all pairs
    """
    N = bbox.shape[0]
    ii, jj = np.meshgrid(range(N), range(N))
    ii, jj = ii.flatten(), jj.flatten()
    is_non_diag = ii != jj  # IoU for diag is always 1.0
    ii, jj = ii[is_non_diag], jj[is_non_diag]
    return ii, jj

def _compute_iou(box_1, box_2, method="iou"):
    """
    Since there are many IoU-like metrics,
    we compute them at once and return the specified one.
    box_1 and box_2 are in (N, 4) format.
    """
    assert method in ["iou", "giou", "ai/a1", "ai/a2"]

    if isinstance(box_1, torch.Tensor):
        box_1 = np.array(box_1)
        box_2 = np.array(box_2)
    assert len(box_1) == len(box_2)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))

    au = a1 + a2 - ai
    # import warnings
    # def handle_warning(message, category, filename, lineno, file=None, line=None):
    #     print("box_1", box_1, "box_2",box_2, "au", au, sep="\n")
    # warnings.showwarning = handle_warning
    # with warnings.catch_warnings():
    #     warnings.simplefilter("always", RuntimeWarning)
    
    iou = ai / (au + 1e-8)
    # iou = ai / au

    if method == "iou":
        return iou
    elif method == "ai/a1":
        return ai / (a1 + 1e-8)
        # return ai / a1
    elif method == "ai/a2":
        return ai / (a2 + 1e-8)
        # return ai / a2

    # outer region
    l_min = np.minimum(l1, l2)
    r_max = np.maximum(r1, r2)
    t_min = np.minimum(t1, t2)
    b_max = np.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou

def iou_func_factory(name: str = "iou"):
    return IOU_FUNC_FACTORY[name]

IOU_FUNC_FACTORY = {
    "iou": partial(_compute_iou, method="iou"),
    "ai/a1": partial(_compute_iou, method="ai/a1"),
    "ai/a2": partial(_compute_iou, method="ai/a2"),
    "giou": partial(_compute_iou, method="giou"),
    # "perceptual": _compute_perceptual_iou,
}

def _get_coords(batch, validate_range=True):
    xc, yc = batch["center_x"], batch["center_y"]
    xl = xc - batch["width"] / 2.0
    xr = xc + batch["width"] / 2.0
    yt = yc - batch["height"] / 2.0
    yb = yc + batch["height"] / 2.0

    if validate_range:
        xl = torch.maximum(xl, torch.zeros_like(xl))
        xr = torch.minimum(xr, torch.ones_like(xr))
        yt = torch.maximum(yt, torch.zeros_like(yt))
        yb = torch.minimum(yb, torch.ones_like(yb))
    return xl, xc, xr, yt, yc, yb

def __compute_underlay_effectiveness(layout, underlay_id):
    """
    Ratio of valid underlay elements to total underlay elements used in PosterLayout.
    Intuitively, underlay should be placed under other non-underlay elements.
    - strict: scoring the underlay as
        1: there is a non-underlay element completely inside
        0: otherwise
    - loose: Calcurate (ai/a2).
    Aggregation part is following the original code (description in paper is not enough).
    Higher values are better (in 0.0 - 1.0 range).
    """
    bbox, label = layout
    N = bbox.shape[0]
    if N in [0, 1]:
        # no overlap in principle
        return {
            "underlay_effectiveness_loose": None,
            "underlay_effectiveness_strict": None,
        }

    ii, jj = _list_all_pair_indices(bbox)
    iou = iou_func_factory("ai/a2")(bbox[ii], bbox[jj])
    mat, mask = np.zeros((N, N)), np.full((N, N), fill_value=False)
    mat[ii, jj] = iou
    mask[ii, jj] = True

    # mask out iou between underlays
    underlay_inds = [i for (i, id_) in enumerate(label) if id_ == underlay_id]
    for i, j in itertools.product(underlay_inds, underlay_inds):
        mask[i, j] = False

    loose_scores, strict_scores = [], []
    for i in range(N):
        if label[i] != underlay_id:
            continue

        score = mat[i][mask[i]]
        if len(score) > 0:
            loose_score = score.max()

            # if ai / a2 is (almost) 1.0, it means a2 is completely inside a1
            # if we can find any non-underlay object inside the underlay, it is ok
            # thresh is used to avoid numerical small difference
            thresh = 1.0 - 1e-5
            strict_score = (score >= thresh).any().astype(np.float32)
        else:
            loose_score = 0.0
            strict_score = 0.0
        loose_scores.append(loose_score)
        strict_scores.append(strict_score)

    return {
        "underlay_effectiveness_loose": _mean(loose_scores),
        "underlay_effectiveness_strict": _mean(strict_scores),
    }

def _mean(values):
    if len(values) == 0:
        return None
    else:
        return sum(values) / len(values)

def _extract_grad(image):
    image_npy = rearrange(np.array(image * 255), "c h w -> h w c")
    image_npy_gray = cv2.cvtColor(image_npy, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(image_npy_gray, -1, 1, 0)
    grad_y = cv2.Sobel(image_npy_gray, -1, 0, 1)
    grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
    # ?: is it really OK to do content adaptive normalization?
    grad_xy = grad_xy / np.max(grad_xy)
    return torch.from_numpy(grad_xy)

# execution functions

def parseResultStr(result_str):
    result_str = result_str.split('\n')
    result_dict = {}
    wanted = ["validity", "overlay", "alignment-LayoutGAN++", "underlay_effectiveness_loose", "underlay_effectiveness_strict", "utilization", "occlusion", "unreadability", "intention_coverage", "intention_conflict"]
    
    for s in result_str:
        try:
            k, v = s.split(': ')
        except:
            continue
        if k in wanted:
            if k in result_dict:
                result_dict[k].append(v)
            else:
                result_dict[k] = [v]
    for i in range(len(result_dict[wanted[0]])):
        print(" & ".join([result_dict[w][i] for w in wanted]) + r" \\")
    
def run(root, base_dir, label_id, format_from, dataset):
    print(root)
    text_id = label_id['text']
    underlay_id = label_id['underlay']

    if format_from == 'ralf':
        file = os.listdir(root)
        file = list(filter(lambda x: x.endswith('.pkl'), file))

        metrics = []
        for pkl_file in file:
            with open(os.path.join(root, pkl_file), 'rb') as f:
                p = pkl.load(f)['results']
            valid_pts, metric_val = compute_validity(p)
            print("validity:", metric_val)
            batch = ptsToBatch(valid_pts)
    
            metric = {}
            # metric.update(compute_overlay(batch, underlay_id=underlay_id))
            # metric.update(compute_alignment(batch))
            # metric.update(compute_underlay_effectiveness(batch, underlay_id=underlay_id))
            # metric.update(compute_saliency_aware_metrics(batch, base_dir, text_id=text_id, underlay_id=underlay_id))
            metric.update(compute_density_aware_metrics(batch, base_dir, dataset=dataset))
            metrics.append(metric)
            
        metric = {}
        for k in metrics[0].keys():
            metric[k] = sum([np.mean(m[k]) for m in metrics]) / len(metrics)
        
        result_str = f"validity: {metric_val}\n"
        for k, v in metric.items():
            result_str += f"{k}: {np.mean(v)}\n"
            print(f"{k}: {np.mean(v)}")
        
        try:
            parseResultStr(result_str)
        except:
            pass  
        
        
    elif format_from == 'layoutprompter':
        for split_name in ['valid', 'test']:
            if split_name == 'valid':
                with open(os.path.join(f"../calg_dataset/{dataset}/test.txt"), 'r') as f:
                    ids = f.read().split('\n')
                file = list(map(lambda x: x + '.pt', ids))
            else:
                ids = read_csv(f"../calg_dataset/{dataset}/annotation/test.csv").poster_path.tolist()
                file = list(map(lambda x: x[:-3] + 'pt', ids))
            
            if dataset == 'pku':
                file = sorted(file, key=lambda x: int(x.split('.')[0]))
            file = list(map(lambda x: os.path.join(root, split_name, x), file))
            print(f"Evaluating the {split_name} split with {len(file)} samples.")
            
            gathered_pts = gather_pt(file)
            valid_pts, metric_val = compute_validity(gathered_pts)
            print("validity:", metric_val)
            # print(valid_pts)
            batch = ptsToBatch(valid_pts)

            metric = {}
            metric.update(compute_overlay(batch, underlay_id=underlay_id))
            metric.update(compute_alignment(batch))
            metric.update(compute_underlay_effectiveness(batch, underlay_id=underlay_id))
            metric.update(compute_saliency_aware_metrics(batch, base_dir[split_name], text_id=text_id, underlay_id=underlay_id))
            metric.update(compute_density_aware_metrics(batch, base_dir[split_name], dataset=dataset))
            
            result_str = f"validity: {metric_val}\n"
            for k, v in metric.items():
                result_str += f"{k}: {np.mean(v)}\n"
                print(f"{k}: {np.mean(v)}")

            try:
                parseResultStr(result_str)
            except:
                pass  
        
    elif format_from == 'postero':
        splits = torch.load(root, weights_only=False)
        for split_name in ['valid', 'test']:
            print(f"Evaluating the {split_name} split with {len(splits[split_name])} samples.")
        
            gathered_split = gather_pSplit(splits[split_name], label_id)
            valid_split, metric_val = compute_validity(gathered_split)
            print("validity:", metric_val)
            batch = ptsToBatch(valid_split)
    
            metric = {}
            metric.update(compute_overlay(batch, underlay_id=underlay_id))
            metric.update(compute_alignment(batch))
            metric.update(compute_underlay_effectiveness(batch, underlay_id=underlay_id))
            metric.update(compute_saliency_aware_metrics(batch, base_dir[split_name], text_id=text_id, underlay_id=underlay_id))
            metric.update(compute_density_aware_metrics(batch, base_dir[split_name], dataset=dataset))

            result_str = f"validity: {metric_val}\n"
            for k, v in metric.items():
                result_str += f"{k}: {np.mean(v)}\n"
                print(f"{k}: {np.mean(v)}")

            try:
                parseResultStr(result_str)
            except:
                pass

def junk(x, s, d, g, e, r):
    return None
import warnings
warnings.showwarning = junk

def run_root_format_dataset(root, format_from, dataset):
    dataset_label = {
        'pku': {1: 'text', 2: 'logo', 3: 'underlay'},
        'cgl': {1: 'logo', 2: 'text', 3: 'underlay', 4: 'embellishment'}
    }
    label_id = {v: k for k, v in dataset_label[dataset].items()}
    
    base_dir = {
        'valid': {'general': f"/home/xuxiaoyuan/calg_dataset/{dataset}/image/train/",
                  'density': "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/train"},
        'test': {'general': f"/home/xuxiaoyuan/calg_dataset/{dataset}/image/test/",
                 'density': "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/test"}
    }
    run(root, base_dir, label_id, format_from, dataset)
    
if __name__ == '__main__':
    root = ""
    format_from = "postero"
    dataset = "pku"
    run_root_format_dataset(root, format_from, dataset)
    