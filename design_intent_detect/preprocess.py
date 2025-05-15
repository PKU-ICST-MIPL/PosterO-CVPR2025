import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pandas import read_csv
import argparse
from tqdm import tqdm

def getClosedDM(dataset_root="../../calg_dataset/", annotation_file_name="train_csv_9973.csv", dataset="pku"):
    path = f"{dataset_root}/{dataset}/image/train/input"
    save = f"{dataset_root}/{dataset}/image/train/closedm"
    os.makedirs(save, exist_ok=True)
    files = os.listdir(path)
    df = read_csv(f"{dataset_root}/{dataset}/annotation/{annotation_file_name}")
    groups = df.groupby(df.poster_path)
    for f in tqdm(files[:20]):
        img = Image.new("L", (513, 750))
        draw = ImageDraw.Draw(img, "L")
        query = f
        boxes = groups.get_group(query).box_elem.values
        boxes = [eval(box) for box in boxes]
        for box in boxes:
            # print(box)
            if box[0] > box[2]:
                box[0], box[2] = box[2], box[0]
            if box[1] > box[3]:
                box[1], box[3] = box[3], box[1]
            draw.rectangle(box, fill="white")
        kernel = np.ones((9,9), np.uint8)
        img = img.resize((240, 350))
        closed_img = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
        Image.fromarray(closed_img).save(os.path.join(save, f))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["pku", "cgl"])
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    getClosedDM(dataset_root=args.dataset_root, annotation_file_name="train.csv", dataset=args.dataset)