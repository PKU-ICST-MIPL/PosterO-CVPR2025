import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from model import design_intent_detector
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from dataloader import closedm
from runner import train, test, get_features
from utils import get_args, set_seed
import os
import re

def main():
    # args, ddp, random seed
    args = get_args()
    if args.local_rank <= 0:
        if args.infer:
            print(f"Experiment name: {args.exp_name}, ckpt name: {args.infer_ckpt}")
        else:
            print(f"Experiment name: {args.exp_name}")
        

    # ddp
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        set_seed(rank=dist.get_rank())
    else:
        set_seed(use_gpu=False)

    # dataset
    pp = get_preprocessing_fn('mit_b1', pretrained='imagenet')
    if args.infer is False:
        train_dataset = closedm(args=args, preprocess_fn=pp, split='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=train_sampler)
    test_dataset = closedm(args=args, preprocess_fn=pp, split='test')
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, sampler=test_sampler)
    test_canvas_df = test_dataset.df

    # model
    action = 'extract' if args.extract else 'forward'
    model_divs = design_intent_detector(act=args.model_dm_act, action=action)
    model_divs = model_divs.to(args.device)
    if args.infer:
        model_divs.load_state_dict(torch.load(args.infer_ckpt, weights_only=True))

    if args.local_rank >= 0:
        model_divs = DDP(model_divs, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # training-related
    optimizer = optim.AdamW(params=model_divs.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss().to(args.device)
    
    os.makedirs(os.path.join(args.exp_name, "ckpt"), exist_ok=True)
    if args.vis_preview:
        os.makedirs(os.path.join(args.exp_name, "vis_preview"), exist_ok=True)
    elif args.extract:
        if args.local_rank <= 0:
            print(f"Extracting {len(test_canvas_df)} samples.")
        args.save_dir = os.path.join(args.exp_name, "result", os.path.split(args.infer_ckpt)[-1][:-4], f"{args.dataset}_features", args.extract_split)
        os.makedirs(args.save_dir, exist_ok=True)
    elif args.infer:
        if args.local_rank <= 0:
            print(f"Inferencing {len(test_canvas_df)} samples.")
        args.save_dir = os.path.join(args.exp_name, "result", os.path.split(args.infer_ckpt)[-1][:-4], args.infer_csv)
        os.makedirs(args.save_dir, exist_ok=True)
    
    # run
    if args.infer is False:
        for e in range(args.epoch):
            args.current_epoch = e
            train(model_divs, args, train_loader, optimizer, criterion)
            if e % 5 == 0:
                test(model_divs, args, test_loader, test_canvas_df)
    elif args.extract:
        args.current_epoch = re.findall("epoch(\d*).pth", args.infer_ckpt)[0]
        get_features(model_divs, args, test_loader, test_canvas_df)
    else:
        args.current_epoch = re.findall("epoch(\d*).pth", args.infer_ckpt)[0]
        test(model_divs, args, test_loader, test_canvas_df)

if __name__ == '__main__':
    main()