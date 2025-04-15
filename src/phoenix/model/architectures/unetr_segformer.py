import warnings
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from transformers import SegformerForSemanticSegmentation

from phoenix.model.architectures.unetr import UNETR
from phoenix.model.architectures.mini_unetr import MiniUNETR


class UNETR_Segformer(nn.Module):
    def __init__(self,
                 epsilon=1e-3,
                 input_dim=1,
                 unetr_out_channels=32,
                 segformer_from_pretrained='nvidia/mit-b5',
                 dropout=0.2,
                 patch_size=None,
                 label_size=None,
                 in_chans=None,
                 verbose=True,
                 mini_unetr=False,
                 **kwargs):
        """
        Initialize UNETR_Segformer with individual parameters.

        Args:
            epsilon: Epsilon value for UNETR
            input_dim: Input dimension
            unetr_out_channels: Output channels for UNETR
            patch_size: Patch size
            segformer_from_pretrained: Pretrained model name for Segformer
            dropout: Dropout rate
            in_chans: Number of input channels
            label_size: Label size
            verbose: Whether to print initialization messages
        """
        super().__init__()

        self.epsilon = epsilon
        self.input_dim = input_dim
        self.unetr_out_channels = unetr_out_channels
        self.segformer_from_pretrained = segformer_from_pretrained
        self.patch_size = patch_size
        self.label_size = label_size
        self.in_chans = in_chans
        self.verbose = verbose

        self.dropout = nn.Dropout2d(dropout)

        self.encoder = UNETR(
            epsilon=epsilon,
            input_dim=input_dim,
            output_dim=unetr_out_channels,
            img_shape=(16, patch_size, patch_size)
        ) if not mini_unetr else MiniUNETR(
            epsilon=epsilon,
            input_dim=input_dim,
            output_dim=unetr_out_channels,
            img_shape=(16, patch_size, patch_size)
        )

        original_tf_logger_level = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()

        # Suppress specific warnings during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

            self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path=segformer_from_pretrained,
                num_channels=unetr_out_channels,
                ignore_mismatched_sizes=True,
                num_labels=1,
            )

        transformers.logging.set_verbosity(original_tf_logger_level)

        # Count parameters in each model component
        unetr_params = sum(p.numel() for p in self.encoder.parameters())
        segformer_params = sum(p.numel() for p in self.encoder_2d.parameters())
        total_params = unetr_params + segformer_params

        print(f"\nModel Parameters:")
        print(f"UNETR parameters: {unetr_params:,}")
        print(f"Segformer parameters: {segformer_params:,}")
        print(f"Total parameters: {total_params:,}")
        print("=" * 50)

    def forward(self, image):
        """
         Forward pass through the model.

         Args:
             image: Input image tensor of shape [batch, channels, depth, height, width]

         Returns:
             torch.Tensor: Output logits
         """
        output = self.encoder(image).max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output).logits.squeeze(1)
        return output

    def print_model_summary(self, checkpoint_path=None):
        """
           Print a summary of the model's configuration.

           Args:
               checkpoint_path: Optional path to checkpoint for reference in the summary
           """
        print("\n" + "=" * 50)
        print("UNETR_Segformer Model Summary")
        print("=" * 50)

        if checkpoint_path:
            print(f"Loaded from: {Path(checkpoint_path).name}")

        print("Model Configuration:")
        print(f"  - in_chans: {self.in_chans}")
        print(f"  - patch_size: {self.patch_size}")
        print(f"  - label_size: {self.label_size}")

        print("\nUNETR Configuration:")
        print(f"  - input_dim: {self.input_dim}")
        print(f"  - output_dim: {self.unetr_out_channels}")
        print(f"  - epsilon: {self.epsilon}")

        print("\nSegformer Configuration:")
        print(f"  - input_dim: {self.unetr_out_channels}")
        print(f"  - output_dim: 1")

        print("=" * 50 + "\n")
