import os
import argparse
import torch
import numpy as np
import random
from utils import *
from torch.nn.functional import interpolate
from model_cls import Net
from model_dis import Discriminator
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='RGFormer Training')
parser.add_argument('--data_root',   type=str, required=True,
                    help='Root directory containing Train/ and Test/ subdirectories')
parser.add_argument('--save_path',   type=str, default='./results')
parser.add_argument('--check_path',  type=str, default=None,
                    help='Path to checkpoint .pt file (optional)')
parser.add_argument('--target_size', type=int, default=256)
parser.add_argument('--batch_size',  type=int, default=32)
parser.add_argument('--max_epoch',   type=int, default=150)
parser.add_argument('--lr',          type=float, default=1e-4)
args = parser.parse_args()


input_size        = 32  
train_target_size = 128
train_input_size  = 64

divides    = {1024: 1, 512: 1, 256: 2, 128: 2, 64: 4, 32: 8}
patch_dict = {32: 4, 64: 8, 128: 8, 256: 16, 512: 16, 1024: 32}
dim_dict   = {32: 64, 64: 128, 128: 128, 256: 256, 512: 256, 1024: 512}
divide     = divides[args.target_size]
## End

img_dir     = os.path.join(args.data_root, 'Train', 'GT_TPS_Ref_1024')
val_img_dir = os.path.join(args.data_root, 'Test',  'GT_TPS_Ref_1024')

os.makedirs(args.save_path, exist_ok=True)
tensorboard_log_dir = os.path.join(args.save_path, 'log')
os.makedirs(tensorboard_log_dir, exist_ok=True)

writer = SummaryWriter(log_dir=tensorboard_log_dir)
global_step = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset     = ImageDataset_prog(img_dir,     args.target_size, lr=input_size // 2)
val_dataset = ImageDataset_prog(val_img_dir, args.target_size, lr=input_size // 2)

train_loader = DataLoader(dataset,     batch_size=args.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

Model = []
Dis   = []
ckpt  = torch.load(args.check_path, map_location=device) if args.check_path else None

for i in range(int(np.log2(args.target_size // input_size)) + 1):
    sz = input_size * 2 ** i
    model = Net(
        image_size=(sz, sz),
        patch_size=(patch_dict[sz], patch_dict[sz]),
        dim=dim_dict[sz],
        depth=6, heads=4,
        mlp_dim=dim_dict[sz] // 4 * 2,
        dropout=0.0
    ).to(device)
    discriminator = Discriminator(img_resolution=sz, img_channels=3, divide=divide).to(device)
    if ckpt is not None:
        try:
            model.load_state_dict(ckpt[f'state_dict_{i}'])
            discriminator.load_state_dict(ckpt[f'dis_state_dict_{i}'])
        except Exception:
            print(f'[Warning] Could not load checkpoint for resolution {sz}')
    model.train()
    discriminator.train()
    Model.append(model)
    Dis.append(discriminator)

criterion = nn.L1Loss()
params_g, params_d = [], []
for model, dis in zip(Model, Dis):
    params_g.extend(model.parameters())
    params_d.extend(dis.parameters())
optimizer_g = optim.Adam(params_g, lr=args.lr, betas=(0., 0.99))
optimizer_d = optim.Adam(params_d, lr=args.lr, betas=(0., 0.99))

best_score = 0
len_step   = len(Model)

for epoch in tqdm(range(args.max_epoch), desc="Epochs"):
    epoch_loss = 0

    for model, dis in zip(Model, Dis):
        model.train()
        dis.train()

    batch_save_folder = os.path.join(args.save_path, f'epoch_{epoch + 1}')
    os.makedirs(batch_save_folder, exist_ok=True)

    for batch_idx, (q_img, v_img, q_img_origin, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path) in enumerate(train_loader):
        q_img, q_mask               = q_img.to(device),        q_mask.to(device)
        v_img, v_mask               = v_img.to(device),        v_mask.to(device)
        q_img_origin, q_mask_origin = q_img_origin.to(device), q_mask_origin.to(device)

        optimizer_d.zero_grad()
        outputs, targets, v_imgs, q_masks, v_masks, t_Dis = [], [], [], [], [], []

        for i, model in enumerate(Model):
            sz = 32 * 2 ** i
            if train_input_size <= sz <= train_target_size:
                pre_k      = interpolate(v_img,          (sz // 2, sz // 2), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                pre_k      = interpolate(pre_k,          (sz, sz),           mode='bicubic', align_corners=False).clamp(0, 1)
                pre_v      = interpolate(v_img,          (sz, sz),           mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                pre_v_mask = (interpolate(v_mask,        (sz, sz),           mode='bicubic', align_corners=False, antialias=True) > 0.5).float()
                pre_q_mask = (interpolate(q_mask_origin, (sz, sz),           mode='bicubic', align_corners=False, antialias=True).clamp(0, 1) > 0.5).float()
                targets.append(interpolate(q_img_origin, (sz, sz),           mode='bicubic', align_corners=False, antialias=True).clamp(0, 1))
                q_masks.append(pre_q_mask)
                v_masks.append(pre_v_mask)
                v_imgs.append(pre_v)
                q_img = model(q_img, pre_k * pre_v_mask, pre_v * pre_v_mask).clamp(0, 1)
                outputs.append(q_img)
                t_Dis.append(Dis[i])
            q_img = interpolate(q_img, (sz * 2, sz * 2), mode='bicubic', align_corners=False).clamp(0, 1)

        loss_d_ori = 0
        for out, tar, v_i, q_m, v_m, discriminator in zip(outputs, targets, v_imgs, q_masks, v_masks, t_Dis):
            loss_d_ori += (F.softplus(-discriminator(torch.cat([tar * q_m, v_i * v_m], 0))).mean()
                         + F.softplus(discriminator(out.detach() * q_m)).mean())
        loss_d = 0.1 * loss_d_ori
        loss_d.backward()
        optimizer_d.step()

        # Generator update
        optimizer_g.zero_grad()
        g_loss, l1loss = 0, 0
        for out, tar, q_m, discriminator in zip(outputs, targets, q_masks, t_Dis):
            g_loss += F.softplus(-discriminator(out * q_m)).mean()
            l1loss += criterion(out, tar)
        total_loss = l1loss + g_loss
        total_loss.backward()
        optimizer_g.step()

        epoch_loss += total_loss.item()

        writer.add_scalar("Train/Loss_total", total_loss.item(), global_step)
        writer.add_scalar("Train/Loss_L1",    l1loss.item(),     global_step)
        writer.add_scalar("Train/Loss_G",     g_loss.item(),     global_step)
        writer.add_scalar("Train/Loss_D",     loss_d_ori.item(), global_step)
        global_step += 1

        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f"Epoch [{epoch+1}/{args.max_epoch}], Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"L1: {l1loss.item():.4f}, G: {g_loss.item()/len_step:.4f}, D: {loss_d_ori.item()/len_step:.4f}")

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{args.max_epoch}], Average Training Loss: {avg_epoch_loss:.4f}")

    for model, dis in zip(Model, Dis):
        model.eval()
        dis.eval()

    # Validation
    with torch.no_grad():
        val_psnr = 0
        for batch_idx, (q_img, v_img, q_img_origin, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path) in enumerate(val_loader):
            q_img, q_mask               = q_img.to(device),        q_mask.to(device)
            v_img, v_mask               = v_img.to(device),        v_mask.to(device)
            q_img_origin, q_mask_origin = q_img_origin.to(device), q_mask_origin.to(device)

            outputs = []
            for i, model in enumerate(Model):
                sz = 32 * 2 ** i
                if train_input_size <= sz <= train_target_size:
                    pre_k      = interpolate(v_img,   (sz // 2, sz // 2), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                    pre_k      = interpolate(pre_k,   (sz, sz),           mode='bicubic', align_corners=False).clamp(0, 1)
                    pre_v      = interpolate(v_img,   (sz, sz),           mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                    pre_v_mask = (interpolate(v_mask, (sz, sz),           mode='bicubic', align_corners=False, antialias=True) > 0.5).float()
                    q_img = model(q_img, pre_k * pre_v_mask, pre_v * pre_v_mask).clamp(0, 1)
                    outputs.append(interpolate(q_img, (train_target_size, train_target_size), mode='bicubic', align_corners=False).clamp(0, 1))
                q_img = interpolate(q_img, (sz * 2, sz * 2), mode='bicubic', align_corners=False).clamp(0, 1)

            out           = outputs[-1]
            q_img_origin  = interpolate(q_img_origin, (train_target_size, train_target_size), mode='bicubic', align_corners=False).clamp(0, 1)
            v_img_resized = interpolate(v_img,        (train_target_size, train_target_size), mode='bicubic', align_corners=False).clamp(0, 1)

            val_psnr += sum(calculate_psnr(out[i], q_img_origin[i]) for i in range(q_img_origin.size(0))) / q_img_origin.size(0)

            if batch_idx == 0:
                save_output_images(torch.cat(outputs, -1), torch.cat([v_img_resized, q_img_origin], -1), batch_save_folder)

    avg_val_psnr = val_psnr / len(val_loader)
    print(f"Epoch [{epoch+1}/{args.max_epoch}], Average Validation PSNR: {avg_val_psnr:.2f} dB")

    writer.add_scalar("Epoch/Loss",            avg_epoch_loss, epoch + 1)
    writer.add_scalar("Epoch/Validation_PSNR", avg_val_psnr,  epoch + 1)

    with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
        f.write(f"{epoch+1},{avg_epoch_loss:.4f},{avg_val_psnr:.2f}\n")

    save_dict = {}
    for i, (model, dis) in enumerate(zip(Model, Dis)):
        save_dict[f"state_dict_{i}"] = model.state_dict()
        save_dict[f"dis_state_dict_{i}"] = dis.state_dict()

    torch.save(save_dict, os.path.join(args.save_path, 'last.pt'))
    if avg_val_psnr > best_score:
        best_score = avg_val_psnr
        torch.save(save_dict, os.path.join(args.save_path, 'best.pt'))

writer.flush()
writer.close()
