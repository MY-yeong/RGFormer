import os
import math
import glob
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


def calculate_psnr(output, target):
    mse = nn.functional.mse_loss(output, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


def get_img(x):
    x = x.clamp(0, 1)
    output_image = x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    return (output_image * 255).astype(np.uint8)


def save_output_images(output_tensors, target_tensors, folder, step=0):
    os.makedirs(folder, exist_ok=True)
    for i, (output_tensor, target_tensor) in enumerate(zip(output_tensors, target_tensors)):
        output = get_img(output_tensor)
        target = get_img(target_tensor)
        result_image = Image.fromarray(np.concatenate([output, target], 1))
        result_image.save(f"{folder}/num_{step + i + 1}.jpg")
    return step + i + 1


class ImageDataset_infer(Dataset):
    def __init__(self, img_dir, size, lr):
        self.img_dir = glob.glob(os.path.join(img_dir, '*_0.*'))
        self.size = size
        self.lr = lr

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        q_img_path = self.img_dir[idx]
        v_mask_fn  = q_img_path.replace('_0.', '_2_mask.')
        v_img_path = q_img_path.replace('_0.', '_2.')

        q_img        = self.preprocess_image(q_img_path, sz=self.lr)
        q_img_origin = self.preprocess_image(q_img_path, sz=self.size, is_LR=False)
        v_img        = self.preprocess_image(v_img_path, sz=self.size, is_LR=False)
        v_mask       = self.preprocess_image(v_mask_fn,  sz=self.size, is_LR=False)

        return q_img, v_img, q_img_origin, v_mask, q_img_path, v_img_path

    def preprocess_image(self, image_path, sz, is_LR=True):
        image = Image.open(image_path)
        if is_LR:
            transform = transforms.Compose([
                transforms.Resize((sz, sz), Image.BICUBIC),
                transforms.Resize((sz * 2, sz * 2), Image.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((sz, sz), Image.BICUBIC),
                transforms.ToTensor(),
            ])
        return transform(image)


class ImageDataset_prog(Dataset):
    def __init__(self, img_dir, size, lr, mask='../facial_mask'):
        self.images = []
        self.masks  = []
        for fn in glob.glob(f'{img_dir}/*_0.jpg'):
            filename = os.path.basename(fn)
            mask_fn  = f'{mask}/{filename}'
            if os.path.exists(mask_fn.replace('_0.jpg', '_2.jpg')) and os.path.exists(mask_fn):
                self.images.append(fn)
                self.masks.append(mask_fn)
        self.size = size
        self.lr   = lr

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        q_img_path  = self.images[idx]
        v_img_path  = q_img_path.replace('_0.jpg', '_2.jpg')
        q_mask_fn   = self.masks[idx]
        v_mask_fn   = q_mask_fn.replace('_0.jpg', '_2.jpg')

        q_img         = self.preprocess_image(q_img_path, sz=self.lr)
        q_img_origin  = self.preprocess_image(q_img_path, sz=self.size, is_LR=False)
        v_img         = self.preprocess_image(v_img_path, sz=self.size, is_LR=False)
        q_mask        = self.preprocess_image(q_mask_fn,  sz=self.lr)
        q_mask_origin = self.preprocess_image(q_mask_fn,  sz=self.size, is_LR=False)
        v_mask        = self.preprocess_image(v_mask_fn,  sz=self.size, is_LR=False)

        return q_img, v_img, q_img_origin, q_mask, v_mask, q_mask_origin, q_img_path, v_img_path

    def preprocess_image(self, image_path, sz, is_LR=True):
        image = Image.open(image_path)
        if is_LR:
            transform = transforms.Compose([
                transforms.Resize((sz, sz), Image.BICUBIC),
                transforms.Resize((sz * 2, sz * 2), Image.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((sz, sz), Image.BICUBIC),
                transforms.ToTensor(),
            ])
        return transform(image)


class ImageDataset(Dataset):
    def __init__(self, img_dir, size):
        self.img_dir  = img_dir
        self.q_images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if '_0.jpg' in f])
        self.k_images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if '_2.jpg' in f])
        self.size     = size

    def __len__(self):
        return len(self.q_images)

    def __getitem__(self, idx):
        q_img_path = self.q_images[idx]
        k_img_path = self.k_images[idx]
        v_img_path = self.k_images[idx]

        q_img        = self.preprocess_image(q_img_path, sz=self.size // 2)
        q_img_origin = self.preprocess_image(q_img_path, sz=self.size, is_LR=False)
        k_img        = self.preprocess_image(k_img_path, sz=self.size // 2)
        v_img        = self.preprocess_image(v_img_path, sz=self.size, is_LR=False)

        return q_img, k_img, v_img, q_img_origin, q_img_path, v_img_path

    def preprocess_image(self, image_path, sz, is_LR=True):
        image = Image.open(image_path)
        if is_LR:
            transform = transforms.Compose([
                transforms.Resize((sz, sz), Image.BICUBIC),
                transforms.Resize((sz * 2, sz * 2), Image.BICUBIC),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((sz, sz), Image.BICUBIC),
                transforms.ToTensor(),
            ])
        return transform(image)