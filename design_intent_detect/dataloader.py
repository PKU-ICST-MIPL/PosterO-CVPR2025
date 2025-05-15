import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from pandas import read_csv

class closedm(Dataset):
    def __init__(self, args, preprocess_fn=None, split="train"):
        assert preprocess_fn, "No preprocess functions provided."
        
        if args.dataset == "all":
            if split == "test":
                self.canvas_dir = os.path.join(args.dataset_root, "{}", "image", args.infer_csv, "input")
                self.closedm_dir = os.path.join(args.dataset_root, "{}", "image", args.infer_csv, "closedm")
                self.df = read_csv(os.path.join(args.dataset_root, "all", "annotation", f"{args.infer_csv}.csv"))
            else:
                self.canvas_dir = os.path.join(args.dataset_root, "{}", "image", split, "input")
                self.closedm_dir = os.path.join(args.dataset_root, "{}", "image", split, "closedm")
                self.df = read_csv(os.path.join(args.dataset_root, "all", "annotation", f"{split}.csv"))
        else:
            if split == "test":
                self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "image", args.infer_csv, "input")
                self.closedm_dir = os.path.join(args.dataset_root, args.dataset, "image", args.infer_csv, "closedm")
                self.df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", f"{args.infer_csv}.csv"))
            else:
                self.canvas_dir = os.path.join(args.dataset_root, args.dataset, "image", split, "input")
                self.closedm_dir = os.path.join(args.dataset_root, args.dataset, "image", split, "closedm")
                self.df = read_csv(os.path.join(args.dataset_root, args.dataset, "annotation", f"{split}.csv"))

        self.transform_canvas = transforms.Compose([
            lambda x: cv2.resize(x, (224, 224)),
            preprocess_fn,
            transforms.ToTensor()
        ])
        if split == "train":
            self.transform_closedm = transforms.Compose([
                lambda x: cv2.resize(x, (224, 224)),
                transforms.ToTensor()
            ])

        if "split" in self.df:
            if args.extract:
                self.df = self.df[self.df["split"] == args.extract_split]
            elif not args.infer:
                self.df = self.df[self.df["split"] != "valid"]
        self.df = self.df.drop_duplicates(subset=['poster_path']).reset_index(drop=True)
        if split == "test" and args.vis_preview:
            self.df = self.df.iloc[:32]
        
        self.use_all = True if args.dataset == "all" else False
        self.train = True if split == "train" else False
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        entry = self.df.iloc[idx]
        if self.use_all:
            ds = str(entry.dataset)
            if self.train:
                closedm = cv2.imread(os.path.join(self.closedm_dir.format(ds), entry.poster_path), 0)
            canvas_path = os.path.join(self.canvas_dir.format(ds), entry.poster_path)
        else:
            if self.train:
                closedm = cv2.imread(os.path.join(self.closedm_dir, entry.poster_path), 0)
            canvas_path = os.path.join(self.canvas_dir, entry.poster_path)
            
        canvas = cv2.imread(canvas_path)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        if self.train:
            return self.transform_canvas(canvas).float(), self.transform_closedm(closedm).float()
        else:
            return self.transform_canvas(canvas).float(), idx