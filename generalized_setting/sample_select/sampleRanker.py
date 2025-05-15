import torch
import torch.nn as nn
from torchvision.ops.boxes import box_area
import numpy as np
import os

def label_similarity(X, Y, penalty=0.2):
    X = sorted(X)
    Y = sorted(Y)
    similarity = 0
    i = 0
    j = 0
    while i < len(X) and j < len(Y):
        if X[i] == Y[j]:
            similarity += 1
            i += 1
            j += 1
        elif X[i] < Y[j]:
            i += 1
            similarity -= penalty
        else:
            j += 1
            similarity -= penalty
    return similarity

def denbox_similarity(boxes1, boxes2):
    if type(boxes1) is not torch.Tensor:
        boxes1 = torch.tensor(boxes1)
    if type(boxes2) is not torch.Tensor:
        boxes2 = torch.tensor(boxes2)
            
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.array(inter) / np.array(union)
    
    return np.mean(iou)

def feature_similarity(feature_src, features_tgt):
    feature_src = feature_src.unsqueeze(0) # (1, 512, 7, 7), while the features_tgt is (N, 512, 7, 7)
    return torch.mean(nn.CosineSimilarity()(feature_src, features_tgt), dim=(1, 2))

class SampleRanker:
    def __init__(self, args, layout_planter, pool_strategy, rank_strategy, self_exclude_train=False):
        '''Rank the pool of samples and select a subset'''
        
        pooler = SamplePooler(pool_strategy)
        self.pool = pooler(args, layout_planter)
        self.sample_size = args.sample_size
        self.self_exclude_train = self_exclude_train
        
        if rank_strategy == 'random':
            self.sampler = self.random_sample
            self.candidates = np.arange(len(self.pool))
        elif rank_strategy == 'rank_by_label':
            self.sampler = self.rank_by_label
            self.candidates = list(enumerate([self.pool[i]['layout']['cls_elem'] for i in range(len(self.pool))]))
        elif rank_strategy == 'rank_by_denbox':
            self.sampler = self.rank_by_denbox
            self.candidates = list(enumerate([self.pool[i]['den_box'] for i in range(len(self.pool))]))
        elif rank_strategy == 'rank_by_feature':
            self.sampler = self.rank_by_feature
            candidates = [os.path.splitext(self.pool[i]['poster_path'])[0] for i in range(len(self.pool))]
            candidates = list(map(lambda x: torch.tensor(np.load(os.path.join(args.feature_dir, "train", x+'.npy'))), candidates))
            self.candidates = list(enumerate(candidates))
            self.feature_dir = args.feature_dir
        else:
            raise ValueError(f'Invalid sample ranker: {rank_strategy}')
        
        self.rank_strategy = rank_strategy
            
    def random_sample(self, **kwargs):
        if self.self_exclude_train:
            selected = self.candidates[:self.sample_size + 1].tolist()
            if kwargs['self_i'] in selected:
                selected.remove(kwargs['self_i'])
            else:
                selected = selected[:self.sample_size]
        else:
            selected = self.candidates[:self.sample_size].tolist()
        return selected
    
    def rank_by_label(self, **kwargs):
        anchor = kwargs['labels']
        selected = [(candidate[0], label_similarity(anchor, candidate[1])) for candidate in self.candidates]
        selected = list(map(lambda x: x[0], sorted(selected, key=lambda x: x[1], reverse=True)[:self.sample_size]))
        return selected
    
    def rank_by_denbox(self, **kwargs):
        anchor = kwargs['instance']['den_box']
        selected = [(candidate[0], denbox_similarity(anchor, candidate[1])) for candidate in self.candidates]
        selected = list(map(lambda x: x[0], sorted(selected, key=lambda x: x[1], reverse=True)[:self.sample_size]))
        return selected
        
    def rank_by_feature(self, **kwargs):
        split_name = kwargs['split_name']
        anchor = kwargs['instance']['poster_path']
        anchor = torch.tensor(np.load(os.path.join(self.feature_dir, split_name, os.path.splitext(anchor)[0] + '.npy')))
        similarity = feature_similarity(anchor, torch.stack([candidate[1] for candidate in self.candidates]))
        selected = list(map(lambda x: x[0], sorted([(self.candidates[i][0], similarity[i]) for i in range(len(self.candidates))], key=lambda x: x[1], reverse=True)[:self.sample_size]))
        return selected
        
    def __call__(self, **kwargs):
        np.random.shuffle(self.candidates)
        sample = self.sampler(**kwargs)
        return {'layout_description': [self.pool[i]['layout_description'] for i in sample],
                'poster_path': [self.pool[i]['poster_path'] for i in sample],
                'label': [self.pool[i]['layout']['cls_elem'] for i in sample]}
        

class SamplePooler:
    def __init__(self, pool_strategy):
        '''Generate a pool of samples before selection'''
        self.pool_strategy = pool_strategy
        
    def load_metric(self, layout_planter, metric_path):
        metric_train = torch.load(metric_path, weights_only=False)
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
        self.metric_train = {v: np.array(metric_train[k]) for k, v in metric_map.items()}
        
        for k, v in self.metric_train.items():
            assert len(v) == len(layout_planter.db_train), f'{k} has different length from db_train'

        for i in range(len(layout_planter.db_train)):
            layout_planter.db_train[i]['metric'] = {k: v[i] for k, v in self.metric_train.items()}
            
    def filterByMetric(self, layout_planter, filter_dict):
        # filter_dict: {metric_name: (min_value, max_value)}
        
        mask = np.ones(len(layout_planter.db_train), dtype=bool)
        for metric_name, (min_value, max_value) in filter_dict.items():
            if metric_name not in self.metric_train:
                raise ValueError(f'Invalid metric name: {metric_name}')
            mask &= (self.metric_train[metric_name] >= min_value) & (self.metric_train[metric_name] <= max_value)
        
        return [layout_planter.db_train[i] for i in range(len(layout_planter.db_train)) if mask[i]]
    
    def describeByMetric(self, layout_planter, describe_list):
        metric_description_map = {'ove': 'Overlay',
                                'ali': 'Alignment',
                                'und_l': 'Underlay Effectiveness (Loose)',
                                'und_s': 'Underlay Effectiveness (Strict)',
                                'uti': 'Non-salient Region Utilization',
                                'occ': 'Salient Region Occlusion',
                                'rea': 'Non-readability',
                                'cov': 'Intention Region Coverage',
                                'con': 'Intention Region Conflict',}
        for i in range(len(layout_planter.db_train)):
            metric_description = ', '.join([f'{metric_description_map[k]}={v:.3f}' for k, v in layout_planter.db_train[i]['metric'].items() if k in describe_list])
            new_description = (layout_planter.db_train[i]['layout_description'][0] + f'It achieves the following metrics: {metric_description}.\n',
                               layout_planter.db_train[i]['layout_description'][1])
            layout_planter.db_train[i]['layout_description'] = new_description
            
        return layout_planter
        
    def clusterByMetric(self, layout_planter, cluster_args):
        pass
    
    def __call__(self, args, layout_planter):
        if self.pool_strategy == 'all':
            return layout_planter.db_train
        elif self.pool_strategy == 'metric_filter':
            self.load_metric(layout_planter, args.metric_path)
            return self.filterByMetric(layout_planter, args.filter_dict)
        elif self.pool_strategy == 'metric_describe':
            self.load_metric(layout_planter, args.metric_path)
            return self.describeByMetric(layout_planter, args.describe_list)
        elif self.pool_strategy == 'metric_filter_describe':
            self.load_metric(layout_planter, args.metric_path)
            layout_planter = self.describeByMetric(layout_planter, args.describe_list)
            return self.filterByMetric(layout_planter, args.filter_dict)
        elif self.pool_strategy == 'metric_cluster':
            raise NotImplementedError
        else:
            raise ValueError(f'Invalid pool strategy: {args.pool_strategy}')