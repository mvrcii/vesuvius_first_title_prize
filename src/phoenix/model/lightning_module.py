import torch
import torch.nn.functional as F
import torchmetrics
import wandb
from lightning import LightningModule
from torch.optim import AdamW
from torchvision.utils import make_grid

from phoenix.model.architectures.unetr_segformer import UNETR_Segformer


class SmoothedCombinedLoss(torch.nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, label_smoothing=0.1):
        """
        Combined loss with label smoothing for more robust training.

        Args:
            dice_weight: Weight for the Dice loss component
            bce_weight: Weight for the BCE loss component
            label_smoothing: Amount of smoothing to apply to the target labels (0.0-1.0)
        """
        super(SmoothedCombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.label_smoothing = label_smoothing

    def _smooth_targets(self, targets):
        """Apply label smoothing to binary targets"""
        # For binary targets, smooth positive labels down and negative labels up
        if self.label_smoothing > 0:
            # Positive labels become (1 - label_smoothing)
            # Negative labels become (label_smoothing)
            return targets * (1 - self.label_smoothing) + (1 - targets) * self.label_smoothing
        return targets

    def forward(self, preds, targets, keep_mask=None):
        """
        Calculate combined loss with label smoothing and optional pixel-wise masking.

        Args:
            preds: model predictions
            targets: Ground truth labels
            keep_mask: Binary mask (1 for pixels to include in loss, 0 to exclude)
                  If None, all pixels are included
        """
        if keep_mask is None:
            # If no mask provided, use all pixels
            keep_mask = torch.ones_like(preds)
        else:
            # Ensure mask is floating point
            keep_mask = keep_mask.float()

        # If mask is empty (all zeros), return zero loss
        if keep_mask.sum() < 1e-8:
            return torch.tensor(0.0, device=preds.device)

        # Apply label smoothing to targets
        smoothed_targets = self._smooth_targets(targets)

        # Calculate masked dice loss with smoothed targets
        intersection = (preds * smoothed_targets * keep_mask).sum()
        dice_inputs_sum = (preds * keep_mask).sum()
        dice_targets_sum = (smoothed_targets * keep_mask).sum()
        dice_coef = (2. * intersection + 1e-8) / (dice_inputs_sum + dice_targets_sum + 1e-8)
        dice = 1 - dice_coef

        # Calculate masked BCE loss with smoothed targets - temporarily cast to float32 for BCE calculation
        with torch.cuda.amp.autocast(enabled=False):
            preds_float32 = preds.float()
            smoothed_targets_float32 = smoothed_targets.float()
            keep_mask_float32 = keep_mask.float()

            bce_per_pixel = F.binary_cross_entropy(
                preds_float32,
                smoothed_targets_float32,
                reduction='none',
            )
            bce = (bce_per_pixel * keep_mask_float32).sum() / (keep_mask_float32.sum() + 1e-8)

        # Combine losses with weights
        return self.dice_weight * dice + self.bce_weight * bce


class UNETR_SF_Module(LightningModule):
    """Expects a cleaned config in dict format"""

    def __init__(self, **kwargs):
        super().__init__()

        # Set all attributes from the config as instance variables
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.save_hyperparameters()
        self.model = UNETR_Segformer(**kwargs)

        bce_weight = getattr(self, 'bce_weight', 0.5)
        # expects prediction probabilities
        self.loss_function = SmoothedCombinedLoss(bce_weight=bce_weight, label_smoothing=self.label_smoothing)

        self.train_step = 0
        self.img_log_freq = getattr(self, 'img_log_freq', {'train': 50, 'val': 10})
        self.img_log_count = getattr(self, 'img_log_count', 2)

        self.val_metrics = torch.nn.ModuleDict({
            'iou': torchmetrics.JaccardIndex(task='binary'),
            'precision': torchmetrics.Precision(task='binary'),
            'recall': torchmetrics.Recall(task='binary'),
            'f1': torchmetrics.F1Score(task='binary')
        })

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Create learning rate scheduler with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.cos_max_epochs,  # Total number of epochs
            eta_min=self.cos_eta_min,  # Minimum learning rate
            verbose=False  # Optional: whether to print logs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_keep = label[:, 1]

        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)

        loss = self.loss_function(y_pred, y_true.float(), y_keep.float())

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        self.train_step += 1

        if self.train_step % self.img_log_freq['train'] == 0:
            self.log_images(y_pred, y_true, y_keep, "Train", self.train_step)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        y_true = label[:, 0]
        y_keep = label[:, 1]

        logits = self.forward(data)
        y_pred = torch.sigmoid(logits)
        y_pred_binary = (y_pred > 0.5).float()

        loss = self.loss_function(y_pred, y_true.float(), y_keep.float())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Apply mask and update metrics
        y_pred_masked = y_pred_binary * y_keep
        y_true_masked = y_true * y_keep

        # Update metrics
        for name, metric in self.val_metrics.items():
            metric(y_pred_masked, y_true_masked)

        # More frequent and diverse validation image logging
        if batch_idx % self.img_log_freq['val'] == 0:
            self.log_images(y_pred, y_true, y_keep, "Validation", self.train_step)

    def on_validation_epoch_end(self):
        metric_dict = {f'val_{name}': metric.compute() for name, metric in self.val_metrics.items()}

        # Log to progress bar and wandb
        for name, value in metric_dict.items():
            prog_bar = name in ['val_iou', 'val_f1']  # Only show IoU and F1 in progress bar
            self.log(name, value, prog_bar=prog_bar, sync_dist=True)

        for metric in self.val_metrics.values():
            metric.reset()

    def log_images(self, y_pred, y_true, y_keep, prefix, step, indices=None):
        """
        Log images to wandb with predictions, ground truth, and keep mask.

        Args:
            y_pred: Predicted segmentation masks
            y_true: Ground truth segmentation masks
            y_keep: Mask indicating which pixels to keep/evaluate
            prefix: Prefix for the log name (e.g., "Train", "Validation")
            step: Current training step
            indices: Specific indices to log. If None, uses default indices.
        """
        if not self.trainer.is_global_zero:
            return

        indices = indices or [0, min(3, len(y_pred) - 1)]

        with torch.no_grad():
            for idx in indices:
                # Concatenate the three images side by side (pred, true, keep)
                combined = torch.cat([y_pred[idx], y_true[idx], y_keep[idx]], dim=1)
                grid = make_grid(combined).detach().cpu()
                wandb.log({f"{prefix} Image {idx}": wandb.Image(grid, caption=f"{prefix} Step {step}")})
