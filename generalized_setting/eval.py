import torch
from PIL import Image
import cairosvg
from io import BytesIO
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse

def get_intent_map(base_dir, path, canvas_size):
    if not os.path.exists(os.path.join(base_dir, path)):
        path = path.replace('.jpg', '.png')
    intent_map = cv2.imread(os.path.join(base_dir, path))
    intent_map = cv2.resize(intent_map, canvas_size)
    intent_map = cv2.cvtColor(intent_map, cv2.COLOR_BGR2GRAY)
    intent_map = intent_map / 255.0
    return intent_map

def get_saliency_map(base_dir, path, canvas_size):
    sal_map_1 = cv2.imread(os.path.join(base_dir, "saliency", path))
    sal_map_2 = cv2.imread(os.path.join(base_dir, "saliency_sub", path))
    sal_map_1 = cv2.resize(sal_map_1, canvas_size)
    sal_map_1 = cv2.cvtColor(sal_map_1, cv2.COLOR_BGR2GRAY)
    sal_map_1 = sal_map_1 / 255.0
    sal_map_2 = cv2.resize(sal_map_2, canvas_size)
    sal_map_2 = cv2.cvtColor(sal_map_2, cv2.COLOR_BGR2GRAY)
    sal_map_2 = sal_map_2 / 255.0
    sal_map = np.maximum(sal_map_1, sal_map_2)
    return sal_map

def run(root, dataset_root):
    standard_metric = {
        'chinese-poem': {'cov': 0.2899433488003709, 'con': 0.07802796390089578, 'uti': 0.14412894781106408, 'occ': 0.03623888480371466},
        'food-menu': {'cov': 0.2738582881442205, 'con': 0.14000491878193386, 'uti': 0.19614014517823564, 'occ': 0.0930296917497653},
        'kind-animals': {'cov': 0.32853749762147466, 'con': 0.12050859341732859, 'uti': 0.21628202857581597, 'occ': 0.04215247741103521},
        'london-subway': {'cov': 0.5645646206887496, 'con': 0.2562342316344511, 'uti': 0.37432570123105297, 'occ': 0.1907388047209451},
        'motivational-quote': {'cov': 0.14043006743412695, 'con': 0.051927014799323316, 'uti': 0.07260140046526346, 'occ': 0.055496284825149414},
        'movie-poster': {'cov': 0.2894049207065874, 'con': 0.16417887618530133, 'uti': 0.1836100415651652, 'occ': 0.1768596303508281},
        'travel-vintage': {'cov': 0.303398123043808, 'con': 0.06521768626991964, 'uti': 0.14463420541928915, 'occ': 0.02054441063926326}
    }

    print(root)
    print(" & ".join(["category", "ove", "cov", "con", "uti", "occ", "int", "sal", "avg"]))
    stylish_category = ['chinese-poem', 'food-menu', 'kind-animals', 'london-subway', 'motivational-quote', 'movie-poster', 'travel-vintage']
    for category in stylish_category:
        intent_base_dir = f"{dataset_root}/PStylish7/{category}/predm_zs/test"
        saliency_base_dir = f"{dataset_root}/PStylish7/{category}/image/test/"
        pt = torch.load(root.format("pstylish7_" + category), weights_only=False)
        metrics = {'ove': [], 'cov': [], 'con': [], 'uti': [], 'occ': []}

        for i in range(len(pt['test'])):
            inst = pt['test'][i]
            svg_str = inst['generated']['svg'][0].strip().split('\n')
            svg_str = [s for s in svg_str if 'polygon' not in s]

            opening, ending = svg_str[0], svg_str[-1]
            elements = []
            underlay = []
            for j in range(1, len(svg_str)-1):
                s = svg_str[j]
                if 'canvas_0' in s:
                    canvas = s
                elif 'underlay' in s:
                    underlay.append(s)
                else:
                    elements.append(s)

            style_str = '''<defs><style type="text/css"><![CDATA[
                    rect { fill: white; }
                    ellipse { stroke: white; stroke-width: 30; }
                    path { stroke: white; stroke-width: 30; }
                    *[id^='canvas'] { fill: black; }
                ]]></style></defs>
            '''
            svg_str.insert(1, style_str)
            svg_str = '\n'.join(svg_str)

            # Compute overlay = (sigma(area of each layer) - area of final render) / image size
            layers = []
            for e in elements:
                ele_svg_str = '\n'.join([opening, style_str, canvas, e, ending])
                try:
                    layers.append(np.array(Image.open(BytesIO(cairosvg.svg2png(bytestring=ele_svg_str))).convert('L')) / 255.0)
                except:
                    pass
                    # print('Error in rendering element:', e)
            if len(layers) == 0:
                # print(f"Warning: no available element.")
                continue
            area_with_overlay = 0
            render = np.zeros_like(layers[0])
            for layer in layers: 
                render += layer
                render = np.clip(render, 0, 1)
                area_with_overlay += np.sum(layer)

            ove = max(0, (area_with_overlay - np.sum(render)) / layers[0].size)
            metrics['ove'].append(ove)

            # Compute complete render = elemental renders + underlay renders
            layers = []
            for e in underlay:
                ele_svg_str = '\n'.join([opening, style_str, canvas, e, ending])
                try:
                    layers.append(np.array(Image.open(BytesIO(cairosvg.svg2png(bytestring=ele_svg_str))).convert('L')) / 255.0)
                except:
                    pass
                    # print('Error in rendering underlay:', e)
            for layer in layers:
                render += layer
                render = np.clip(render, 0, 1)

            element_map = render.copy()

            pp = inst['poster_path']
            canvas_size = inst['canvas_size']
            if element_map.shape[::-1] != canvas_size:
                # print(f"Warning: canvas size not match, resize element map {element_map.shape[::-1]} to canvas size {canvas_size}")
                element_map = cv2.resize(element_map, canvas_size)
            intent_map = get_intent_map(intent_base_dir, pp, canvas_size)
            inv_intent_map = 1 - intent_map
            saliency_map = get_saliency_map(saliency_base_dir, pp, canvas_size)
            inv_saliency_map = 1 - saliency_map

            cov = np.sum(intent_map * element_map) / np.sum(intent_map)
            con = np.sum(inv_intent_map * element_map) / np.sum(inv_intent_map)
            uti = np.sum(inv_saliency_map * element_map) / np.sum(inv_saliency_map)
            occ = np.sum(saliency_map * element_map) / np.sum(saliency_map)

            metrics['cov'].append(cov)
            metrics['con'].append(con)
            metrics['uti'].append(uti)
            metrics['occ'].append(occ)
        
        metrics = {k: np.mean(v) for k, v in metrics.items()}
        metrics['int'] = abs(metrics['cov'] - standard_metric[category]['cov'])/(1 - standard_metric[category]['cov']) + \
                        abs(standard_metric[category]['con'] - metrics['con'])/standard_metric[category]['con']
        metrics['sal'] = abs(metrics['uti'] - standard_metric[category]['uti'])/(1 - standard_metric[category]['uti']) + \
                        abs(standard_metric[category]['occ'] - metrics['occ'])/standard_metric[category]['occ']
        metrics['avg'] = (metrics['ove'] + metrics['int'] + metrics['sal']) / 3

        print(" & ".join([category, str(metrics['ove']), str(metrics['cov']), str(metrics['con']), str(metrics['uti']), str(metrics['occ']),
                        str(metrics['int']), str(metrics['sal']), str(metrics['avg'])]))
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--pt_path", type=str, required=True)
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    run(args.pt_path, args.dataset_root)