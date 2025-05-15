import torch
import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import os

def getRegions(design_intent_map, pad, kernel_n, draw=False, draw_img=None):
    if draw:
        assert draw_img, "draw_img is required to draw."
    if pad:
        design_intent_map = np.pad(design_intent_map[5:-5, 5:-5], 10, mode='constant', constant_values=0)
    threshold_value = design_intent_map.mean()
    _, binary = cv2.threshold(design_intent_map, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = (kernel_n, kernel_n)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones(kernel, np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(kernel, np.uint8))
    edges = cv2.Canny(binary, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        # print("Rectangle vertices:", [(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        bbox.append((x, y, x+w, y+h))

    if draw:
        draw = ImageDraw.Draw(draw_img)
        for r in bbox:
            draw.rectangle(r, outline="blue", width=5)
        return Image.fromarray(binary), draw_img, bbox
    else:
        return None, None, bbox
        
def getDesignIntentBox(dm_root, csv_dir, split, subsplit=None, pad=False, kernel_n=49, preview=10, canvas_root=None):
    if preview > 0:
        assert canvas_root, "canvas_root is required to preview."
    dm_dir = os.path.join(dm_root, split)
    # files = list(map(lambda x: os.path.join(dm_dir, x), os.listdir(dm_dir)))
    # files = list(filter(lambda x: x.endswith('.png'), files))
    # files = files[:1]
    df = read_csv(os.path.join(csv_dir, f"{split}.csv"))
    if subsplit:
        df = df[df['split'] == subsplit]
    df = df.drop_duplicates(subset=['poster_path']).reset_index(drop=True)

    print(f"Start inferencing {len(df)} samples of {split + str(subsplit).replace('None', '')}.")
    
    if 'dataset' not in df.columns:
        if 'pku' in csv_dir:
            df['dataset'] = 'pku'
        elif 'cgl' in csv_dir:
            df['dataset'] = 'cgl'
        else:
            raise ValueError("Dataset not found.")
    
    save_bbox = []
    for i in range(len(df)):
        entry = df.iloc[i]
        dm = cv2.imread(os.path.join(dm_dir, f"{entry.dataset}_{entry.poster_path}"), 0)
        min_val = np.min(dm)
        max_val = np.max(dm)
        scale_factor = 255 / (max_val - min_val)
        dm = dm * scale_factor
        dm = dm.astype(np.uint8)

        if i < preview:
            draw_img = Image.open(os.path.join(canvas_root, entry.dataset, "image", split, "input", entry.poster_path)).resize((513, 750))
            binary, drawn, bbox = getRegions(dm, pad, kernel_n, draw=True, draw_img=draw_img)
            print(i)
            plt.subplot(1, 3, 1)
            plt.imshow(Image.fromarray(dm).convert("RGB"))
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(binary)
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(drawn)
            plt.axis("off")
            plt.show()
        else:
            _, _, bbox = getRegions(dm, pad, kernel_n)

        dict_bbox = {'idx_in_all': i, 'den_box': bbox, 'dataset': entry.dataset, 'poster_path': entry.poster_path}
        save_bbox.append(dict_bbox)

    if subsplit:
        split = subsplit
    torch.save(save_bbox, os.path.join(dm_root, f"design_intent_bbox_{split}.pt"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset", default='pku', type=str, choices=['pku', 'cgl'])
    parser.add_argument('--dm_root', type=str, default='')
    parser.add_argument('--infer_ckpt', type=str, default='')
    parser.add_argument('--pad', type=bool, default=False)
    parser.add_argument('--kernel_n', type=int, default=37)
    args = parser.parse_args()
    
    assert args.dm_root or args.infer_ckpt, "dm_root or infer_ckpt is required."
    assert not(args.dm_root and args.infer_ckpt), "Only one of dm_root or infer_ckpt is required, get dm_root: {} and infer_ckpt: {}.".format(args.dm_root, args.infer_ckpt)
    
    if args.infer_ckpt:
        args.dm_root = args.infer_ckpt.replace('ckpt', 'result').replace('.pth', '')
        
    args.csv_dir = os.path.join(args.dataset_root, args.dataset, 'annotation')
    return args

def main():
    args = get_args()
    
    # test
    _ = getDesignIntentBox(
        args.dm_root, 
        args.csv_dir,
        split='test',
        pad=args.pad,
        kernel_n=args.kernel_n,
        preview=0
    )

    # valid
    _ = getDesignIntentBox(
        args.dm_root, 
        args.csv_dir,
        split='train',
        subsplit='valid',
        pad=args.pad,
        kernel_n=args.kernel_n,
        preview=0
    )

    # train
    _ = getDesignIntentBox(
        args.dm_root, 
        args.csv_dir,
        split='train',
        subsplit='train',
        pad=args.pad,
        kernel_n=args.kernel_n,
        preview=0
    )
    
if __name__ == '__main__':
    main()
    