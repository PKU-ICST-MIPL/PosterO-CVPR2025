import torch
import torch.distributed as dist
import numpy as np
import os
from utils import distributed_concat, continuityLoss
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import time

def train(model, args, loader, optimizer, criterion):
    model.train()
    loader.sampler.set_epoch(args.current_epoch)
    epoch_mse_loss = 0
    batch_mse_loss = 0
    if args.use_con_loss:
        epoch_con_loss = 0
        batch_con_loss = 0
    if args.local_rank <= 0:
        print(f"Start training @ Epoch {args.current_epoch} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).")
    for b, (canvas, closedm) in enumerate(tqdm(loader)):
        canvas, closedm = canvas.to(args.device), closedm.to(args.device)
        optimizer.zero_grad()
        predm = model(canvas)
        mse_loss = criterion(predm, closedm)
        loss = mse_loss
        if args.use_con_loss:
            con_loss = continuityLoss(predm, closedm)
            loss += con_loss
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            mse_loss = mse_loss.clone().detach()
            if args.local_rank >= 0:
                dist.reduce(mse_loss, dst=0)
            if args.local_rank <= 0:
                batch_mse_loss += mse_loss
                epoch_mse_loss += mse_loss
                if args.use_con_loss:
                    batch_con_loss += con_loss
                    epoch_con_loss += con_loss
                if args.local_rank == 0:
                    batch_mse_loss /= dist.get_world_size()
                    if args.use_con_loss:
                        batch_con_loss /= dist.get_world_size()
                if b % 10 == 0:
                    if args.use_con_loss:
                        print(f"Epoch {args.current_epoch} | Batch {b} | MSELoss: {batch_mse_loss:.4f}, CONLoss: {batch_con_loss:.4f}")
                        batch_con_loss = 0
                    else:
                        print(f"Epoch {args.current_epoch} | Batch {b} | Loss: {batch_mse_loss:.4f}")
                    batch_mse_loss = 0
    with torch.no_grad():
        if args.local_rank <= 0:
            epoch_mse_loss /= dist.get_world_size()
            epoch_mse_loss /= len(loader)
            if args.use_con_loss:
                epoch_con_loss /= dist.get_world_size()
                epoch_con_loss /= len(loader)
                print(f"[ Epoch {args.current_epoch} ] AvgMSELoss: {epoch_mse_loss:.4f}, AvgCONLoss: {epoch_con_loss:.4f}")
            else:
                print(f"[ Epoch {args.current_epoch} ] AvgLoss: {epoch_mse_loss:.4f}")

def test(model, args, loader, canvas_df, dont_save_for_speed_test=False):
    model.eval()
    total_time = 0
    
    with torch.no_grad():
        predms = []
        cvidc = []
        if args.local_rank <= 0:
            print(f"Start testing @ Epoch {args.current_epoch} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).")
        for canvas, cvidx in tqdm(loader):
            canvas = canvas.to(args.device)
            cvidx = cvidx.to(args.device)
            start_time = time.time()
            predm = model(canvas)
            end_time = time.time()
            total_time += end_time - start_time
            predms.append(predm)
            cvidc.append(cvidx)

        if args.infer:
            if not args.vis_preview:
                predms = torch.concat(predms, dim=0)
                cvidc = torch.concat(cvidc, dim=0)
                # print(predms.shape, cvidc.shape)
                if dont_save_for_speed_test:
                    pass
                else:
                    save_maps(args, predms, canvas_df, cvidc)

        if args.vis_preview:
            predms = distributed_concat(torch.concat(predms, dim=0), 
                                            len(canvas_df))
            cvidc = distributed_concat(torch.concat(cvidc, dim=0),
                                            len(canvas_df))
        
        if args.local_rank <= 0:
            if args.vis_preview:
                visualize(args, predms, canvas_df, cvidc)
            if not args.infer:
                root = os.path.join(args.exp_name, "ckpt")
                torch.save(model.module.state_dict(), os.path.join(root, f"epoch{args.current_epoch}.pth"))
                
    return total_time
                
def get_features(model, args, loader, canvas_df):
    model.eval()
    with torch.no_grad():
        features = []
        cvidc = []
        if args.local_rank <= 0:
            print(f"Start getting {args.dataset}-{args.extract_split} features @ Epoch {args.current_epoch} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).")
        for canvas, cvidx in tqdm(loader):
            canvas = canvas.to(args.device)
            cvidx = cvidx.to(args.device)
            features.append(model(canvas))
            cvidc.append(cvidx)
            
        features = torch.concat(features, dim=0)
        cvidc = torch.concat(cvidc, dim=0)
        print(features.shape, cvidc.shape)
        for c, f in zip(cvidc.detach().cpu().numpy(), features.detach().cpu().numpy()):
            pp = canvas_df.iloc[c].poster_path
            np.save(os.path.join(args.save_dir, f"{os.path.splitext(os.path.basename(pp))[0]}"), f)

def save_maps(args, predms, cvdf, cvidc):
    for _, (predm, cvidx) in enumerate(zip(predms.detach().cpu().numpy(), cvidc.detach().cpu().numpy())):
        entry = cvdf.iloc[cvidx]
        if 'dataset' in entry:
            save_name = f"{entry.dataset}_{entry.poster_path}"
        else:
            save_name = f"{args.dataset}_{entry.poster_path}"
        img = Image.fromarray(predm.squeeze(0) * 255).resize((513, 750)).convert("RGB")
        img.save(os.path.join(args.save_dir, save_name))


def visualize(args, predms, cvdf, cvidc):
    plt.figure(figsize=(14, 16))
    root = os.path.join(args.exp_name, "vis_preview")

    for idx, (predm, cvidx) in enumerate(zip(predms.detach().cpu().numpy(), cvidc.detach().cpu().numpy())):
        entry = cvdf.iloc[cvidx]
        if 'dataset' in entry:
            cvpath = os.path.join(args.dataset_root, entry.dataset, "image", "test", "input", entry.poster_path)
        else:
            cvpath = os.path.join(args.dataset_root, args.dataset, "image", "test", "input", entry.poster_path)
        cv = Image.open(cvpath).convert("RGB")
        plt.subplot(8, 8, 2 * idx + 1)
        plt.axis("off")
        plt.imshow(cv)
        plt.subplot(8, 8, 2 * idx + 2)
        plt.axis("off")
        plt.imshow(Image.fromarray(predm.squeeze(0) * 255).resize((513, 750)))
    plt.tight_layout()

    plt.savefig(os.path.join(root, f"Epoch{args.current_epoch}.png"))