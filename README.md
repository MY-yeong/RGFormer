# RGFormer: Reference-Guided Transformer for Face Super-Resolution (Neurocomputing 2026)

[![Paper](https://img.shields.io/badge/Paper-Neurocomputing-blue)](https://doi.org/10.1016/your-doi-here)

> Min-Yeong Kim\*, Seung-Wook Kim\*, and Keunsoo Ko, "Reference-Guided Transformer for Face Super-Resolution"
>
> \*These authors contributed equally to this work.

---

## Requirements

```bash
# 1. Clone repository
git clone https://github.com/keunsoo-ko/RGFormer.git
cd RGFormer

# 2. Create anaconda environment
conda create -n rgformer python=3.9 -y
conda activate rgformer

# 3. Install PyTorch
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install CodeFormer
git clone https://github.com/sczhou/CodeFormer.git
cd CodeFormer
pip install -r requirements.txt
cd ..
```

---

## Pretrained Weights

Download the pretrained weights from [Google Drive](https://drive.google.com/your-link-here) and place them in the `weights/` directory.

```
weights/
└── best.pt
```

---

## Dataset Preparation

```
dataset/
├── LR/
│   ├── 000001.jpg
│   └── ...
├── Ref/
│   ├── 000001.jpg
│   └── ...
└── Mask/
    ├── 000001.jpg
    └── ...
```

---

## Training

```bash
python train.py \
    --lq_dir /path/to/LR \
    --ref_dir /path/to/Ref \
    --mask_dir /path/to/Mask \
    --save_path ./results \
    --batch_size 32 \
    --max_epoch 150
```

## Inference

```bash
python infer.py \
    --lq_dir /path/to/LR \
    --ref_dir /path/to/Ref \
    --mask_dir /path/to/Mask \
    --check_path ./weights/best.pt \
    --save_path ./infer_results
```

---

## Testset

Download test pairs of target and reference: [Google Drive](https://drive.google.com/drive/folders/1HaQEJR6lApgd5E0d4ybzq-ExCaCiX5GA?usp=drive_link)

---
## Qualitative comparison on the Wider dataset
<img width="2089" height="1147" alt="image" src="https://github.com/user-attachments/assets/42219456-46b5-4f0c-8ed2-cbdabef3fcdb" />

---

## Acknowledgements

This work builds upon [CodeFormer](https://github.com/sczhou/CodeFormer). We thank the authors for their excellent work.
