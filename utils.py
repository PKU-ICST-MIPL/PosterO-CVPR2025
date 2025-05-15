import torch
import random
import numpy as np
import argparse
import os
import json
import time

def set_seed(seed=42, use_gpu=True, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_args_infer_dataset():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset_name", default='pku', type=str, choices=['pku', 'cgl'])
    parser.add_argument("--design_intent_bbox_dir", type=str, required=True)
    parser.add_argument("--annotation_dir", type=str, required=True)
    parser.add_argument("--ext_save_name", type=str, default='')
    parser.add_argument("--canvas_size_w", type=int, default=513)
    parser.add_argument("--canvas_size_h", type=int, default=750)
    parser.add_argument("--structure", type=str, nargs='*')
    parser.add_argument("--injection", type=str, nargs='*')
    
    # hyperpm
    parser.add_argument("--exp_name", default='', type=str)
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    
    # divs split model
    # parser.add_argument("--model_divs_ckpt", default='', type=str)

    # large language model
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--max_tokens", default=800, type=int)
    parser.add_argument("--top_p", default=1, type=int)
    parser.add_argument("--frequency_penalty", default=0, type=int)
    parser.add_argument("--presence_penalty", default=0, type=int)
    parser.add_argument("--num_return", default=10, type=int)
    parser.add_argument("--stop_token", default="\n\n", type=str)
    
    # sample & ranker
    parser.add_argument("--pool_strategy", default='all', type=str, choices=['all', 'metric_filter', 'metric_describe', 'metric_filter_describe', 'metric_cluster'])
    parser.add_argument("--rank_strategy", default='random', type=str, choices=['random', 'rank_by_label', 'rank_by_denbox', 'rank_by_feature'])
    parser.add_argument("--sample_size", required=True, type=int)
    
    parser.add_argument("--metric_path", default='', type=str)
    parser.add_argument("--filter_dict", default='', type=str)
    parser.add_argument("--describe_list", default='', type=str, nargs='*', choices=['ove', 'ali', 'und_l', 'und_s', 'uti', 'occ', 'rea', 'cov', 'con'])
    parser.add_argument("--feature_dir", default='', type=str)
    
    parser.add_argument("--label_rback", action='store_true')
    
    # bool
    parser.add_argument("--vis_preview", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--save_rag", action='store_true')

    args = parser.parse_args()
    
    if args.dataset_name == 'pku':
        label_info = {
            1: {'type': 'text', 'color': 'green'}, 
            2: {'type': 'logo', 'color': 'red'},
            3: {'type': 'underlay', 'color': 'orange'}
        }
    elif args.dataset_name == 'cgl':
        label_info = {
            1: {'type': 'logo', 'color': 'red'},
            2: {'type': 'text', 'color': 'green'},
            3: {'type': 'underlay', 'color': 'orange'},
            4: {'type': 'embellishment', 'color': 'blue'}
        }
        
    args.dataset_info = {
        'dataset_name': args.dataset_name,
        'design_intent_bbox_dir': args.design_intent_bbox_dir,
        'annotation_dir': args.annotation_dir,
        'label_info': label_info
    }
    
    if args.ext_save_name == '':
        args.ext_save_name = None
    args.canvas_size = (args.canvas_size_w, args.canvas_size_h)
    
    # dealing sampleRanker
    if args.pool_strategy.startswith('metric'):
        assert args.metric_path != '', "Please provide metric_path."
        assert args.filter_dict != '', "Please provide filter_dict."
        if args.pool_strategy.endswith('describe'):
            assert args.describe_list != '', "Please provide describe_list."

        with open(args.filter_dict, 'r') as f:
            filter_dict = json.load(f)
        assert filter_dict['dataset'] == args.dataset_name, f"provided filter_dict is not compatible to {args.dataset_name}."
        args.filter_dict = filter_dict['metric']
        
    if args.rank_strategy == 'rank_by_feature':
        assert args.feature_dir != '', "Please run extrach.sh first and provide feature_dir."
    
    if args.exp_name == '':
        args.exp_name = time.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        args.exp_name = f"{args.exp_name}_{args.N}"
    
    save_dir = os.path.join(os.path.split(args.model_dir)[-1], args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    args.save_path = os.path.join(save_dir, f"{{}}_{{}}_{args.exp_name}.pt")
    
    return args