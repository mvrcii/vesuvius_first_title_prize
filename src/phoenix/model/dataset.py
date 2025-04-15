import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize


class UNETR_SF_Dataset(Dataset):
    def __init__(self, root_dir, images, transform, cfg, labels=None, mode="train", preload=True):
        self.cfg = cfg
        self.images = np.array(images)
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.label_shape = (2, cfg.label_size, cfg.label_size)
        self.label_shape_upscaled = (2, cfg.patch_size, cfg.patch_size)
        self.patch_size = cfg.patch_size
        self.preload = preload

        # Preload data into memory with preprocessing
        if self.preload:
            self.preloaded_images = {}
            self.preloaded_labels = {}
            print(f"Preloading {len(self.images)} samples into memory...")
            for i, (img_path, label_path) in enumerate(zip(self.images, self.labels)):

                img_full_path = os.path.join(self.root_dir, img_path)
                label_full_path = os.path.join(self.root_dir, label_path)

                # Load image and do pre-processing
                image = np.load(img_full_path)

                # For images: transpose to (H,W,C) for augmentation later
                image = np.transpose(image, (1, 2, 0))
                self.preloaded_images[img_path] = image

                # Load and preprocess label
                label = np.load(label_full_path)

                # Unpack bits and reshape label
                label = np.unpackbits(label).reshape(self.label_shape)

                # Scale label up to patch shape - do this in preloading
                label = resize(label, self.label_shape_upscaled, order=0, preserve_range=True, anti_aliasing=False)

                # Transpose to (H,W,C) for augmentation later
                label = np.transpose(label, (1, 2, 0))

                self.preloaded_labels[label_path] = label

            print("Preloading complete!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load pre-processed data from memory
        if self.preload:
            image = self.preloaded_images[self.images[idx]]
            label = self.preloaded_labels[self.labels[idx]]
        else:
            # Only if not preloaded, do the full processing
            image = np.load(os.path.join(self.root_dir, self.images[idx]))
            label = np.load(os.path.join(self.root_dir, self.labels[idx]))

            # Unpack and reshape label
            label = np.unpackbits(label).reshape(self.label_shape)

            # Scale label up to patch shape
            label = resize(label, self.label_shape_upscaled, order=0, preserve_range=True, anti_aliasing=False)

            # Rearrange for augmentations
            image = np.transpose(image, (1, 2, 0))
            label = np.transpose(label, (1, 2, 0))

        # Apply augmentations and normalization (can't precompute)
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']

        # Post-augmentation processing
        # Rearrange image back from (H,W,C) to (C,H,W)
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        # Scale label back down to label shape (only done after augmentation)
        label = resize(label, self.label_shape, order=0, preserve_range=True, anti_aliasing=False)
        label = torch.tensor(label, dtype=torch.float16)

        # Go from (layers, patch_size, patch_size) to (1, layers, patch_size, patch_size)
        image = torch.tensor(image).unsqueeze(0)

        # Pad image to have 16 layers
        image = torch.cat([image, torch.zeros(1, 16 - image.shape[1], self.patch_size, self.patch_size)], dim=1)

        return image, label
