import torch
import os
import numpy as np
from metrics import compute_validity, compute_alignment, compute_overlay, compute_underlay_effectiveness, \
                    compute_saliency_aware_metrics, compute_density_aware_metrics, \
                    ptsToBatch, gather_pSplit
                    
def get_postero_result_metrics(postero_result_path, layout_planter_path, base_dir):
    planter = torch.load(layout_planter_path, weights_only=False)
    label_id, underlay_id, text_id, dataset = get_infos_from_planter(planter)
    
    splits = torch.load(postero_result_path, weights_only=False)
    for split_name in ['valid', 'test']:
        data = gather_pSplit(splits[split_name], label_id)
        print(len(data))
    
        new_data, val = compute_validity(data)
        batch = ptsToBatch(new_data)
        
        get_metric(batch, base_dir[split_name], text_id, underlay_id, dataset, f'metric_result_{split_name}_{os.path.split(postero_result_path)[-1]}')
                    
def get_db_train_metrics(layout_planter_path, base_dir):
    planter = torch.load(layout_planter_path, weights_only=False)
    label_id, underlay_id, text_id, dataset = get_infos_from_planter(planter)

    data = []
    for p in planter['db_train']:
        box = torch.tensor(p['layout']['box_elem'], dtype=torch.float64)
        box[:, ::2] = box[:, ::2] / 513.0
        box[:, 1::2] = box[:, 1::2] / 750.0
        cls = torch.tensor(p['layout']['cls_elem'], dtype=int)
        
        d = {
            'center_x': (box[:, 0] + box[:, 2]) / 2,
            'center_y': (box[:, 1] + box[:, 3]) / 2,
            'width': box[:, 2] - box[:, 0],
            'height': box[:, 3] - box[:, 1],
            'id': os.path.splitext(p['poster_path'])[0],
            'label': cls
        }
        data.append(d)
    print(len(data))
    
    new_data, val = compute_validity(data)
    batch = ptsToBatch(new_data)
    
    get_metric(batch, base_dir, text_id, underlay_id, dataset, f'metric_train_{planter["dataset_info"]["dataset_name"]}.pt')

def get_infos_from_planter(planter):
    label_id = {v['type']: k for k, v in planter['dataset_info']['label_info'].items()}
    underlay_id = label_id['underlay']
    text_id = label_id['text']
    
    dataset = planter['dataset_info']['dataset_name']
    
    return label_id, underlay_id, text_id, dataset

def get_metric(batch, base_dir, text_id, underlay_id, dataset, save_path):
    
    metric = {}
    metric.update(compute_overlay(batch, underlay_id=underlay_id))
    metric.update(compute_alignment(batch))
    metric.update(compute_underlay_effectiveness(batch, underlay_id=underlay_id))
    metric.update(compute_saliency_aware_metrics(batch, base_dir, text_id=text_id, underlay_id=underlay_id))
    metric.update(compute_density_aware_metrics(batch, base_dir, dataset=dataset))
    
    torch.save(metric, save_path)
    
    metric_map = {
        'overlay': 'ove',
        'alignment-LayoutGAN++': 'ali',
        'underlay_effectiveness_loose': 'und_l',
        'underlay_effectiveness_strict': 'und_s',
        'utilization': 'uti',
        'occlusion': 'occ',
        'unreadability': 'rea',
        'intention_coverage': 'cov',
        'intention_conflict': 'con',
    }
    
    for k, v in metric.items():
        print(metric_map[k], len(v))
        
def get_metric_train():
    # Note: This takes a long time to run, about 25 minutes for pku and 100 minutes for cgl datasets.
    layout_planter_path = "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/pku_preprocessed_dbs_hierarchical_top.pt"
    base_dir = {
        "general": "/home/xuxiaoyuan/calg_dataset/pku/image/train/",
        "density": "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/train"
    }
    get_db_train_metrics(layout_planter_path, base_dir)
    
    layout_planter_path = "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/cgl_preprocessed_dbs_hierarchical_top.pt"
    base_dir = {
        "general": "/home/xuxiaoyuan/calg_dataset/cgl/image/train/",
        "density": "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/train"
    }
    get_db_train_metrics(layout_planter_path, base_dir)

if __name__ == '__main__':
    # get_metric_train()
    dataset = 'pku'
    
    poster_paths = ['/home/xuxiaoyuan/PosterO/Meta-Llama-3.1-8B/pku/hierarchical_top_filter_ove_0.001_cov0.2_con0.2_small_1.pt']
    layout_planter_path = f"/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/{dataset}_preprocessed_dbs_hierarchical_top.pt"
    base_dir = {
        'valid': {'general': f"/home/xuxiaoyuan/calg_dataset/{dataset}/image/train/",
                  'density': "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/train"},
        'test': {'general': f"/home/xuxiaoyuan/calg_dataset/{dataset}/image/test/",
                 'density': "/home/xuxiaoyuan/PosterO/divs_split/all_128_1e-06_none/result/epoch25/test"}
    }
    
    for poster_path in poster_paths:
        get_postero_result_metrics(poster_path,
                                   layout_planter_path,
                                   base_dir)