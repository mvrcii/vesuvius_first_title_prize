import os
import albumentations as A

data_root_dir = 'data'
dataset_target_dir = 'data/scroll5/datasets/ft_on_03192025'

# Architecture
model_type = 'b3'
architecture = 'unetr-sf'
segformer_from_pretrained = f'nvidia/mit-{model_type}'
model_name = 'unetr-sf-b3'
mini_unetr = True
unetr_out_channels = 32
in_chans = 16
patch_size = 128
label_size = 32
stride = 64

# Trainer
epochs = -1
node = True
num_workers = 16
seed = 7340043
val_interval = 1
gradient_clip_val = 1

# Data
scroll_id = 5
contrasted = True
val_frac = 0.05
dataset_fraction = 1
take_full_dataset = False
ink_ratio = 5
no_ink_sample_percentage = 0.75
train_batch_size = 32
val_batch_size = 64
fragment_ids = [
    "03192025/parts_contrasted/03192025_01",
    "03192025/parts_contrasted/03192025_03",
    "03192025/parts_contrasted/03192025_04",
    "03192025/parts_contrasted/03192025_09",
    "03192025/parts_contrasted/03192025_11",
    "03192025/parts_contrasted/03192025_13",
    "03192025/parts_contrasted/03192025_15",
    "03192025/parts_contrasted/03192025_19",
    "03192025/parts_contrasted/03192025_20",
    "03192025/parts_contrasted/03192025_21",
    "03192025/parts_contrasted/03192025_24",
    "03192025/parts_contrasted/03192025_25",
    "03192025/parts_contrasted/03192025_26",
    "03192025/parts_contrasted/03192025_27",
    "03192025/parts_contrasted/03192025_28",
    "03192025/parts_contrasted/03192025_29",
]
validation_fragments = []

# Optimizer
weight_decay = 0.001
label_smoothing = 0.1

# Learning rate schedule
lr = 2e-04
epsilon = 0.001
cos_eta_min = 1e-06
cos_max_epochs = 50

# Augmentations
train_aug = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=360, p=0.5),
    A.Perspective(scale=(0.03, 0.03), p=0.1),
    A.GridDistortion(p=0.1),
    A.Blur(blur_limit=3, p=0.1),
    A.GaussNoise(p=0.1),
    A.RandomResizedCrop(size=(patch_size, patch_size), scale=(0.5, 1.0), ratio=(0.75,1.333), p=0.15),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=360, p=0.1),
    A.RandomGamma(p=0.15, gamma_limit=(30, 80)),
    A.RandomBrightnessContrast(p=0.15, brightness_limit=(-0.2, 0.4), contrast_limit=(-0.2, 0.2)),
    A.Normalize(mean=(0,0,0), std=(1,1,1))
]
val_aug = [
    A.Normalize(mean=(0,0,0), std=(1,1,1))
]
