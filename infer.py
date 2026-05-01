import os
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate

from utils import *
from model_cls import Net
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY


PRETRAIN_MODEL_URL = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

patch_dict = {32: 4, 64: 8, 128: 8, 256: 16, 512: 16, 1024: 32}
dim_dict   = {32: 64, 64: 128, 128: 128, 256: 256, 512: 256, 1024: 512}


parser = argparse.ArgumentParser(description='RGFormer Inference')

# Paths
parser.add_argument('--lq_dir',     type=str, required=True,
                    help='Directory containing LR input images')
parser.add_argument('--ref_dir',    type=str, required=True,
                    help='Directory containing reference images (same filename as LR)')
parser.add_argument('--mask_dir',   type=str, required=True,
                    help='Directory containing reference mask images (same filename as LR)')
parser.add_argument('--check_path', type=str, required=True,
                    help='Path to RGFormer checkpoint .pt file')
parser.add_argument('--save_path',  type=str, default='./infer_results')

# Settings
parser.add_argument('--target_size', type=int,   default=256)
parser.add_argument('--output_size', type=int,   default=512)
parser.add_argument('--batch_size',  type=int,   default=1)
parser.add_argument('--fidelity_w',  type=float, default=0.7,
                    help='CodeFormer fidelity weight (0=quality, 1=fidelity)')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_dataset = ImageDataset_infer(
    lq_dir=args.lq_dir,
    ref_dir=args.ref_dir,
    mask_dir=args.mask_dir,
    size=args.target_size,
    lr=16
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# RGFormer
Model = []
ckpt  = torch.load(args.check_path, map_location=device)

for i in range(int(np.log2(args.target_size // 32)) + 1):
    sz = 32 * 2 ** i
    model = Net(
        image_size=(sz, sz),
        patch_size=(patch_dict[sz], patch_dict[sz]),
        dim=dim_dict[sz],
        depth=6, heads=4,
        mlp_dim=dim_dict[sz] // 4 * 2,
        dropout=0.0
    ).to(device)
    try:
        model.load_state_dict(ckpt[f'state_dict_{i}'])
    except Exception:
        print(f'[Warning] Could not load checkpoint for resolution {sz}')
    model.eval()
    Model.append(model)

# CodeFormer 
codeformer = ARCH_REGISTRY.get('CodeFormer')(
    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
    connect_list=['32', '64', '128', '256']
).to(device)

ckpt_path = load_file_from_url(
    url=PRETRAIN_MODEL_URL['restoration'],
    model_dir='weights/CodeFormer',
    progress=True, file_name=None
)
cf_ckpt = torch.load(ckpt_path, map_location='cpu')['params_ema']
codeformer.load_state_dict(cf_ckpt)
codeformer.eval()


with torch.no_grad():
    for lq_img, ref_img, ref_mask, lq_path in tqdm(test_loader):
        lq_img   = lq_img.to(device)
        ref_img  = ref_img.to(device)
        ref_mask = ref_mask.to(device)

        # RGFormer progressive inference
        q_img = lq_img
        outputs = []
        for i, model in enumerate(Model):
            sz = 32 * 2 ** i
            pre_k      = interpolate(ref_img,  (sz // 2, sz // 2), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            pre_k      = interpolate(pre_k,    (sz, sz),           mode='bicubic', align_corners=False).clamp(0, 1)
            pre_v      = interpolate(ref_img,  (sz, sz),           mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
            pre_v_mask = (interpolate(ref_mask,(sz, sz),           mode='bicubic', align_corners=False, antialias=True) > 0.5).float()
            q_img = model(q_img, pre_k * pre_v_mask, pre_v * pre_v_mask).clamp(0, 1)
            outputs.append(interpolate(q_img, (args.target_size, args.target_size), mode='bicubic', align_corners=False).clamp(0, 1))
            q_img = interpolate(q_img, (sz * 2, sz * 2), mode='bicubic', align_corners=False).clamp(0, 1)

        out = outputs[-1]
        out = interpolate(out, (args.output_size, args.output_size), mode='bicubic', align_corners=False).clamp(0, 1)
        out = out * 2 - 1

        # CodeFormer refinement
        restored = codeformer(out, w=args.fidelity_w, adain=True)[0]
        restored = (restored.clamp(-1, 1) + 1) / 2

        for img, path in zip(restored, lq_path):
            result    = Image.fromarray(get_img(img))
            save_name = os.path.splitext(os.path.basename(path))[0]
            result.save(os.path.join(args.save_path, f'{save_name}.png'))

print(f'Restoration complete')