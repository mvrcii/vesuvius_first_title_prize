import os
import albumentations as A
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from phoenix.model.dataset import UNETR_SF_Dataset
from phoenix.utility.configs import Config
from phoenix.utility.utils import get_frag_name_from_id


class UNETR_SF_DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_img_paths = None
        self.t_label_paths = None
        self.v_img_paths = None
        self.v_label_paths = None
        self.generate_dataset(cfg=cfg)

    def generate_dataset(self, cfg: Config):
        """
        Generate training and validation datasets with balanced samples.

        Args:
            cfg: Configuration object containing dataset parameters
        """
        def generate_file_path(row, frag_id_column='frag_id', filename_column='filename'):
            return os.path.join(get_frag_name_from_id(row[frag_id_column]).upper(), 'images', row[filename_column])

        # Load dataset infos
        csv_path = os.path.join(cfg.dataset_target_dir, 'label_infos.csv')
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset infos: {e}")

        # Validate dataset info csv structure
        required_columns = ['frag_id', 'filename', 'ink_p']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        # Set random seed consistently
        random_state = None if cfg.seed == -1 else cfg.seed

        # Apply ignore threshold if specified
        if getattr(cfg, "max_ignore_th", False) and "ignore_p" in df.columns:
            before_count = len(df)
            df = df[df["ignore_p"] < cfg.max_ignore_th]
            print(f"Ignoring patches with ignore_p > {cfg.max_ignore_th}: {before_count} → {len(df)} samples")

        # Balance dataset if needed
        if not getattr(cfg, "take_full_dataset", False):
            # First, shuffle the dataframe to avoid selection bias
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

            # Count samples in each category
            count_zero = (df['ink_p'] == 0).sum()
            count_greater_than_zero = (df['ink_p'] > 0).sum()
            print(f"Before balancing: {count_zero} samples with ink_p = 0 and {count_greater_than_zero} samples with ink_p > 0")

            # Select ink samples above threshold
            df_ink = df[df['ink_p'] >= cfg.ink_ratio]

            # Select appropriate number of no-ink samples
            no_ink_sample_count = int(len(df_ink) * cfg.no_ink_sample_percentage)
            df_no_ink = df[df['ink_p'] == 0].sample(n=min(no_ink_sample_count, count_zero), random_state=random_state)

            # Combine and shuffle again
            df = pd.concat([df_ink, df_no_ink]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Report final distribution
        count_zero = (df['ink_p'] == 0).sum()
        count_ink = (df['ink_p'] >= cfg.ink_ratio).sum()
        print(f"Final count: {len(df)} samples")
        print(f"\tink_p = 0: {count_zero} samples\n\tink_p > {cfg.ink_ratio}: {count_ink} samples")

        # Generate file paths
        df['file_path'] = df.apply(generate_file_path, axis=1)

        # Validate fragment IDs
        train_fragments = [frag for frag in getattr(cfg, 'fragment_ids', [])]
        validation_fragments = [frag for frag in getattr(cfg, 'validation_fragments', [])]

        if not train_fragments:
            raise ValueError("Training fragments not specified or empty")

        # Check if validation fragments are specified
        if not validation_fragments:
            print("No validation fragments specified, using 15% of training data for validation.")
            # Process all train fragments together
            train_val_df = df[df['frag_id'].isin(train_fragments)]

            # Shuffle the dataframe for random split
            train_val_df = train_val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

            # Calculate split point (15% for validation)
            val_size = int(len(train_val_df) * cfg.val_frac)

            # Split the dataframe
            valid_df = train_val_df.iloc[:val_size]
            train_df = train_val_df.iloc[val_size:]

            print(f"Split {len(train_val_df)} samples into {len(train_df)} training and {len(valid_df)} validation samples")
        else:
            # Original logic when validation fragments are specified
            # Check for fragment overlap
            overlap = set(train_fragments).intersection(set(validation_fragments))
            if overlap:
                print(f"Warning: Fragments {overlap} appear in both training and validation sets")

            # Split by fragments
            train_df = df[df['frag_id'].isin(train_fragments)]
            valid_df = df[df['frag_id'].isin(validation_fragments)]

        # Ensure non-empty datasets
        if len(train_df) == 0:
            raise ValueError("No training samples found for the specified fragments")
        if len(valid_df) == 0:
            raise ValueError("No validation samples found after splitting")

        # Apply dataset fraction if needed
        if cfg.dataset_fraction < 1.0:
            print(f"Taking {cfg.dataset_fraction * 100:.1f}% of the dataset for training and validation")
            train_df = train_df.sample(frac=cfg.dataset_fraction, random_state=random_state)
            valid_df = valid_df.sample(frac=cfg.dataset_fraction, random_state=random_state)

        # Report fragment distribution
        print("\nTraining fragment distribution:")
        print(train_df['frag_id'].value_counts().sort_index())
        print("\nValidation fragment distribution:")
        print(valid_df['frag_id'].value_counts().sort_index())

        # Extract paths
        train_image_paths = train_df['file_path'].tolist()
        val_image_paths = valid_df['file_path'].tolist()

        train_label_paths = [path.replace('images', 'labels') for path in train_image_paths]
        val_label_paths = [path.replace('images', 'labels') for path in val_image_paths]

        print(f"Total train samples: {len(train_image_paths)}")
        print(f"Total validation samples: {len(val_image_paths)}")

        self.t_img_paths = train_image_paths
        self.t_label_paths = train_label_paths
        self.v_img_paths = val_image_paths
        self.v_label_paths = val_label_paths

    def get_transforms(self, dataset_type):
        if dataset_type == 'train':
            transforms = self.cfg.train_aug
            return A.Compose(transforms=transforms, is_check_shapes=False)
        elif dataset_type == 'val':
            transforms = self.cfg.val_aug
            return A.Compose(transforms=transforms, is_check_shapes=False)
        return None

    def build_dataloader(self, dataset_type):
        """
        Build dataloader for training or validation.

        Args:
            dataset_type: Either 'train' or 'val'

        Returns:
            DataLoader: Configured PyTorch DataLoader
        """
        # Select paths based on dataset type
        is_train = dataset_type == 'train'
        images_list = self.t_img_paths if is_train else self.v_img_paths
        label_list = self.t_label_paths if is_train else self.v_label_paths

        # Create dataset
        dataset = UNETR_SF_Dataset(
            cfg=self.cfg,
            root_dir=os.path.join(self.cfg.dataset_target_dir),
            images=images_list,
            labels=label_list,
            transform=self.get_transforms(dataset_type=dataset_type),
            mode=dataset_type
        )

        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=self.cfg.train_batch_size if is_train else self.cfg.val_batch_size,
            shuffle=is_train,  # Only shuffle training data
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def train_dataloader(self):
        return self.build_dataloader(dataset_type='train')

    def val_dataloader(self):
        return self.build_dataloader(dataset_type='val')

    def info(self):
        """
        Print dimensions of images and labels from train and validation datasets.
        """
        if self.t_img_paths is None or self.v_img_paths is None:
            print("Dataset not initialized yet. Call generate_dataset first.")
            return

        # Create temporary dataloaders with batch size 1
        for dataset_type in ['train', 'val']:
            print(f"{dataset_type.upper()} Dataset:")
            temp_loader = self.build_dataloader(dataset_type)
            try:
                batch = next(iter(temp_loader))

                if isinstance(batch, dict):
                    for key, item in batch.items():
                        if isinstance(item, torch.Tensor):
                            print(f"{dataset_type} {key} shape: {item.shape}")
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                    print(f"Image shape: {images.shape}, Image type: {images.dtype}")
                    print(f"Label shape: {labels.shape}, Label type: {labels.dtype}")
                else:
                    print(f"Unknown batch format for {dataset_type} dataset")

            except StopIteration:
                print(f"No samples in {dataset_type} dataset")
            except Exception as e:
                print(f"Error inspecting {dataset_type} dataset: {str(e)}")
