#!/usr/bin/env python
import os
import sys
import argparse
import glob
import numpy as np
import torch
import cv2
import json
import albumentations as A
import time
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from phoenix.model.lightning_module import UNETR_SF_Module
from phoenix.utility.configs import Config


def load_model(checkpoint_path):
    print(f"Loading model checkpoint from {os.path.dirname(checkpoint_path)}...")
    start_time = time.time()

    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_file = next((f for f in os.listdir(checkpoint_dir) if f.endswith('.py')), None)
    config_path = os.path.join(checkpoint_dir, config_file) if config_file else None

    # Add checkpoint directory to path
    sys.path.append(os.path.dirname(checkpoint_dir))

    # Load model from checkpoint file
    model = UNETR_SF_Module.load_from_checkpoint(checkpoint_path, strict=True)

    # Extract hparams from model and create Config
    config = Config.load_from_dict(model.hparams)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    return model, config


def create_weight_mask_round(patch_size):
    y, x = np.mgrid[0:patch_size, 0:patch_size]
    center = patch_size // 2
    sigma = patch_size // 4
    weight_mask = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return weight_mask / weight_mask.max()


def load_single_layer(layer_path, crop_coords=None):
    """Load a single layer with optional cropping"""
    if os.path.exists(layer_path):
        layer = cv2.imread(layer_path, cv2.IMREAD_UNCHANGED)
        if layer.dtype == np.uint16:
            layer = cv2.normalize(layer, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if crop_coords:
            x1, y1, x2, y2 = crop_coords
            layer = layer[y1:y2, x1:x2]

        return layer
    return None


def find_layer_path(layers_dir, layer_idx):
    extensions = ["tif", "png", "jpg"]

    formats = [
        str(layer_idx),           # unpadded: "0", "1", "2"...
        f"{layer_idx:02d}"        # zero-padded to 2 digits: "00", "01", "02"...
    ]

    # Try all combinations of formats and extensions
    for fmt in formats:
        for ext in extensions:
            path = os.path.join(layers_dir, f"{fmt}.{ext}")
            if os.path.exists(path):
                return path

    # No matching file found
    return None


def load_layers(fragment_path, start_layer, end_layer, crop_coords=None, num_workers=8):
    """Load layers using parallel processing"""
    print(f"Loading layers {start_layer} to {end_layer}...")
    start_time = time.time()

    layers_dir = os.path.join(fragment_path, 'layers')
    if not os.path.exists(layers_dir):
        raise FileNotFoundError(f"Layers directory not found: {layers_dir}")

    layer_paths = []
    for idx in range(start_layer, end_layer + 1):
        path = find_layer_path(layers_dir, idx)
        if path:
            layer_paths.append(path)
        else:
            raise FileNotFoundError(f"No image found for layer {idx} (.tif or .png or .jpg)")

    # Use multiprocessing to load layers in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(partial(load_single_layer, crop_coords=crop_coords), layer_paths))

    # Find first valid layer to use as template for missing layers
    template = next((layer for layer in results if layer is not None), None)
    if template is None:
        raise FileNotFoundError(f"No valid layers found between {start_layer} and {end_layer}")

    # Replace None with zero arrays
    layers = [layer if layer is not None else np.zeros_like(template) for layer in results]

    load_time = time.time() - start_time
    print(f"Loaded {len(layers)} layers in {load_time:.2f} seconds")

    return layers


def load_single_layer_efficient(fragment_path, layer_idx, crop_coords=None):
    """Load a single layer efficiently for incremental loading"""
    layers_dir = os.path.join(fragment_path, 'layers')
    path = find_layer_path(layers_dir, layer_idx)

    if not path:
        raise FileNotFoundError(f"No image found for layer {layer_idx} (.tif or .png or .jpg)")

    return load_single_layer(path, crop_coords)


def apply_augmentation(patch_tensor, aug_type):
    """Apply a specific augmentation to a patch tensor"""
    if aug_type == "original":
        return patch_tensor
    elif aug_type == "horizontal_flip":
        return torch.flip(patch_tensor, dims=[-1])
    elif aug_type == "vertical_flip":
        return torch.flip(patch_tensor, dims=[-2])
    elif aug_type == "rotate90":
        return torch.rot90(patch_tensor, k=1, dims=[-2, -1])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def reverse_augmentation(pred_tensor, aug_type):
    """Reverse the augmentation on a prediction tensor"""
    if aug_type == "original":
        return pred_tensor
    elif aug_type == "horizontal_flip":
        return torch.flip(pred_tensor, dims=[-1])
    elif aug_type == "vertical_flip":
        return torch.flip(pred_tensor, dims=[-2])
    elif aug_type == "rotate90":
        return torch.rot90(pred_tensor, k=3, dims=[-2, -1])
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def prepare_patch_batch(patch_indices, patch_coords, layers, patch_size, normalizer, tta_level, batch_size):
    """Prepare a batch of patches for the specified indices"""
    # Define the augmentations to apply based on TTA level
    augmentations = ["original"]
    if tta_level >= 1:
        augmentations.append("horizontal_flip")
    if tta_level >= 2:
        augmentations.append("vertical_flip")
    if tta_level >= 3:
        augmentations.append("rotate90")

    # Create a list to store all patches with their metadata
    all_patches = []

    # Process each patch, creating all required TTA versions
    for patch_idx in patch_indices:
        y, x = patch_coords[patch_idx]

        # Extract patches from each layer
        layer_patches = []
        for layer in layers:
            patch = layer[y:y + patch_size, x:x + patch_size]
            # Apply normalization
            normalized = normalizer(image=patch)['image']
            layer_patches.append(normalized)

        # Convert to tensor - shape is [depth, height, width]
        patch_tensor = torch.tensor(np.stack(layer_patches), dtype=torch.float32)

        # Pad to 16 layers if needed
        if patch_tensor.shape[0] < 16:
            padding = torch.zeros(16 - patch_tensor.shape[0], patch_size, patch_size)
            patch_tensor = torch.cat([patch_tensor, padding], dim=0)

        # Add channel dimension: [depth, height, width] -> [1, depth, height, width]
        patch_tensor = patch_tensor.unsqueeze(0)

        # For each augmentation type, create a tuple (tensor, patch_idx, aug_type)
        for aug_type in augmentations:
            augmented_tensor = apply_augmentation(patch_tensor, aug_type)
            all_patches.append((augmented_tensor, patch_idx, aug_type))

    # Create batches of the specified size
    batches = []
    for i in range(0, len(all_patches), batch_size):
        batch = all_patches[i:i + batch_size]
        batches.append(batch)

    return batches


def run_inference_for_layer(current_start_layer, fragment_path, stride=1, tta_level=0,
                            batch_size=4, crop_coords=None, use_weight_mask=True, num_workers=None,
                            patches_in_ram=10000, mask=None, model=None, prev_layers_data=None, end_layer=None):
    """Run inference on a 12-layer window starting at current_start_layer"""
    # Track individual window inference time
    window_start_time = time.time()

    # Calculate end layer (always 12 layers)
    if "parts" in fragment_path:
        current_end_layer = current_start_layer + 15  # 12 layers inclusive
    else:
        current_end_layer = current_start_layer + 11  # 12 layers inclusive

    in_chans = 16 if "parts" in fragment_path else 12

    print(f"\n{'='*80}")
    print(f"Running inference for window starting at layer {current_start_layer} (layers {current_start_layer}-{current_end_layer}) ending at {end_layer}")
    print(f"{'='*80}")

    # If we have previous layers, and we're just moving one layer forward, reuse them
    if prev_layers_data is not None:
        prev_layers, prev_start_layer = prev_layers_data

        if len(prev_layers) == in_chans and current_start_layer == prev_start_layer + 1:
            print(f"Efficiently reusing {len(prev_layers) - 1} layers from previous window")
            # Remove the first layer (oldest) and add the new one (most recent)
            new_layer_idx = current_end_layer
            start_load_time = time.time()
            new_layer = load_single_layer_efficient(fragment_path, new_layer_idx, crop_coords)
            print(f"Loaded new layer {new_layer_idx} in {time.time() - start_load_time:.2f} seconds")

            # Shift window: remove first layer and append new one
            layers = prev_layers[1:] + [new_layer]
            print(f"Reused layers {current_start_layer}-{current_end_layer-1} and loaded new layer {new_layer_idx}")
    else:
        # Load all layers with parallel processing
        print(f"Loading all layers for initial window or non-consecutive window")
        layers = load_layers(fragment_path, current_start_layer, current_end_layer, crop_coords, num_workers=num_workers)

    # Get patch size from model config
    patch_size = model.hparams.patch_size

    # Create weight mask for patch center weighting or use uniform weights
    if use_weight_mask:
        weight_mask = create_weight_mask_round(patch_size)
    else:
        weight_mask = np.ones((patch_size, patch_size), dtype=np.float32)

    # Prepare output arrays
    output = np.zeros_like(mask, dtype=np.float32)
    weight_sum = np.zeros_like(mask, dtype=np.float32)

    # Calculate stride in pixels
    stride_px = patch_size // stride if stride > 0 else patch_size

    # Generate patch coordinates
    coords_start_time = time.time()
    patch_coords = []
    for y in range(0, mask.shape[0] - patch_size + 1, stride_px):
        for x in range(0, mask.shape[1] - patch_size + 1, stride_px):
            # Skip patches that are completely black in mask
            if np.any(mask[y:y + patch_size, x:x + patch_size] > 0):
                patch_coords.append((y, x))

    total_patches = len(patch_coords)
    print(f"Generated {total_patches} patch coordinates in {time.time() - coords_start_time:.2f} seconds")

    # Calculate number of TTA versions per patch
    tta_versions = 1 + min(tta_level, 3)

    # Calculate number of phases needed
    patches_per_phase = min(patches_in_ram, total_patches)
    num_phases = math.ceil(total_patches / patches_per_phase)

    print(f"TTA level: {tta_level} (using {tta_versions} versions per patch)")
    print(f"Total patches: {total_patches}")
    print(f"Processing will be done in {num_phases} phase(s)")

    # Create normalizer
    normalizer = A.Normalize(mean=(0,0,0), std=(1,1,1))

    # Process in phases
    start_idx = 0

    for phase in range(num_phases):
        phase_start_time = time.time()

        # Calculate indices for this phase
        end_idx = min(start_idx + patches_per_phase, total_patches)
        phase_indices = list(range(start_idx, end_idx))

        print(f"\nPhase {phase+1}/{num_phases}: Processing patches {start_idx} to {end_idx-1} ({len(phase_indices)} patches)")

        # Prepare batches for this phase
        batch_prep_time = time.time()
        tta_batches = prepare_patch_batch(
            phase_indices, patch_coords, layers, patch_size, normalizer, tta_level, batch_size
        )
        print(f"Prepared {len(tta_batches)} TTA batches in {time.time() - batch_prep_time:.2f} seconds")

        # Storage for patch predictions for this phase
        patch_predictions = {}  # Dictionary to store predictions for each patch_idx

        # Run inference on batches for this phase
        inference_start_time = time.time()
        print(f"Running inference on {len(tta_batches)} batches...")

        device = next(model.parameters()).device

        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):  # Use mixed precision for faster inference
            for batch_idx, batch in enumerate(tqdm(tta_batches, desc=f"Phase {phase+1} Inference",
                                                   unit="batch", bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
                # Unpack the batch
                tensors, patch_indices, aug_types = zip(*batch)

                # Stack tensors for batch processing
                stacked_batch = torch.stack([t.squeeze(0) for t in tensors], dim=0).to(device)

                # Make sure tensor is in the right shape
                if stacked_batch.ndim == 4:  # [batch, depth, height, width]
                    stacked_batch = stacked_batch.unsqueeze(1)  # Add channel dimension

                # Run inference
                with torch.no_grad():
                    preds = torch.sigmoid(model(stacked_batch))

                # Process predictions
                for i, (pred, patch_idx, aug_type) in enumerate(zip(preds, patch_indices, aug_types)):
                    # Reverse the augmentation
                    reversed_pred = reverse_augmentation(pred, aug_type)

                    # Add to the predictions dictionary
                    if patch_idx not in patch_predictions:
                        patch_predictions[patch_idx] = []
                    patch_predictions[patch_idx].append(reversed_pred.cpu())

                # Delete batch data to free memory
                del stacked_batch, preds
                torch.cuda.empty_cache()

        inference_time = time.time() - inference_start_time
        print(f"Phase {phase+1} inference completed in {inference_time:.2f} seconds")

        # Accumulate results for this phase
        accumulation_start_time = time.time()
        for patch_idx, preds in patch_predictions.items():
            y, x = patch_coords[patch_idx]

            # Average the predictions for this patch
            avg_pred = torch.stack(preds).mean(dim=0).squeeze().numpy()

            # upscale by 4
            avg_pred = np.kron(avg_pred, np.ones((4, 4)))

            # Apply weight mask and accumulate
            output[y:y + patch_size, x:x + patch_size] += avg_pred * weight_mask
            weight_sum[y:y + patch_size, x:x + patch_size] += weight_mask

        print(f"Accumulated phase {phase+1} results in {time.time() - accumulation_start_time:.2f} seconds")

        # Clear memory for next phase
        del patch_predictions, tta_batches
        torch.cuda.empty_cache()

        # Update for next phase
        start_idx = end_idx

        print(f"Phase {phase+1} completed in {time.time() - phase_start_time:.2f} seconds")

    # Normalize by weights
    valid_mask = weight_sum > 0
    output[valid_mask] /= weight_sum[valid_mask]

    window_time = time.time() - window_start_time
    print(f"Window inference for layers {current_start_layer}-{current_end_layer} completed in {window_time:.2f} seconds")

    # Create metadata for this window
    window_metadata = {
        "start_layer": current_start_layer,
        "end_layer": current_end_layer,
        "inference_time_seconds": window_time
    }

    # Return the output, metadata, and the current layers with start index for potential reuse
    return output, window_metadata, (layers, current_start_layer)


def run_sliding_window_inference(fragment_path, checkpoint_dir, start_layer, end_layer,
                                 stride=1, tta_level=0, batch_size=4, crop_coords=None,
                                 use_weight_mask=True, num_workers=None, patches_in_ram=10000, output_dir=None):
    """Run inference on multiple overlapping windows of 12 layers each"""
    # Track overall time
    total_start_time = time.time()

    # Create base output directory structure
    if output_dir is not None:
        # Use provided output directory
        fragment_name = os.path.basename(fragment_path)
        base_output_dir = os.path.join(output_dir, fragment_name)
    else:
        # Use the original method with timestamp
        base_output_dir = fragment_path.replace("data", "results")
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        if crop_coords is not None:
            timestamp = f"crop_{timestamp}"
        base_output_dir = os.path.join(base_output_dir, timestamp)

    npy_dir = os.path.join(base_output_dir, "npy_files")
    vis_dir = os.path.join(base_output_dir, "visualizations")

    # Create directories if they don't exist
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Set number of workers based on CPU count if not specified
    if num_workers is None:
        num_workers = min(os.cpu_count(), 16)  # Limit to 16 workers max

    print(f"Starting sliding window inference with {num_workers} workers and batch size {batch_size}")

    # Find mask file
    mask_start_time = time.time()
    mask_files = glob.glob(os.path.join(fragment_path, '*mask.png'))
    if not mask_files:
        mask_path = fragment_path.replace("_contrasted", "")
        mask_files = glob.glob(os.path.join(mask_path, '*mask.png'))
        if not mask_files:
            raise FileNotFoundError(f"No mask file found in {fragment_path}")
    mask_path = mask_files[0]

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded mask in {time.time() - mask_start_time:.2f} seconds")

    # Apply crop if specified
    if crop_coords:
        x1, y1, x2, y2 = crop_coords
        mask = mask[y1:y2, x1:x2]

    # Load model (once for all windows)
    model, config = load_model(checkpoint_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Create base run metadata
    run_metadata = {
        "fragment_path": fragment_path,
        "checkpoint_dir": checkpoint_dir.split('/')[2],
        "overall_start_layer": start_layer,
        "overall_end_layer": end_layer,
        "stride": stride,
        "tta_level": tta_level,
        "patch_size": config.patch_size,
        "crop_coords": crop_coords,
        "use_weight_mask": use_weight_mask,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "patches_in_ram": patches_in_ram,
        "start_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "windows": [],
        "completed": False
    }

    # Save initial metadata
    metadata_path = os.path.join(base_output_dir, 'run_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(run_metadata, f, indent=2)

    # Track previously loaded layers for efficient reuse
    prev_layers_data = None

    # Process each window
    for current_start_layer in range(start_layer, end_layer + 1):
        # Run inference for this window, potentially reusing previous layers
        output, window_metadata, current_layers_data = run_inference_for_layer(
            current_start_layer,
            fragment_path,
            stride,
            tta_level,
            batch_size,
            crop_coords,
            use_weight_mask,
            num_workers,
            patches_in_ram,
            mask,
            model,
            prev_layers_data,
            end_layer
        )

        # Save NPY output
        npy_path = os.path.join(npy_dir, f"segmentation_layer_{current_start_layer}.npy")
        np.save(npy_path, output)

        # Save visualization
        vis_path = os.path.join(vis_dir, f"segmentation_layer_{current_start_layer}.png")
        cv2.imwrite(vis_path, (output * 255).astype(np.uint8))

        # Update run metadata
        run_metadata["windows"].append(window_metadata)

        # Save updated metadata after each window
        with open(metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)

        print(f"Saved results for window starting at layer {current_start_layer}")

        # Store current layers data for potential reuse in the next iteration
        prev_layers_data = current_layers_data

    # Finalize metadata
    total_time = time.time() - total_start_time
    run_metadata["total_runtime_seconds"] = total_time
    run_metadata["end_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_metadata["completed"] = True

    # Save final metadata
    with open(metadata_path, 'w') as f:
        json.dump(run_metadata, f, indent=2)

    print(f"Sliding window inference completed.")
    print(f"Processed {end_layer - start_layer + 1} windows")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Results saved to {base_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='3D to 2D Segmentation Sliding Window Inference')
    parser.add_argument('fragment_path', help='Path to the fragment directory')
    parser.add_argument('checkpoint_dir', help='Path to the checkpoint directory')
    parser.add_argument('--start_layer', type=int, required=True, help='Start layer index for first window')
    parser.add_argument('--end_layer', type=int, required=False, help='Start layer index for last window')
    parser.add_argument('--stride', type=int, default=2, help='Stride factor (1=full patch, 2=half patch)')
    parser.add_argument('--tta_level', type=int, default=1, choices=[0, 1, 2, 3],
                        help='TTA level (0=none, 1=horizontal, 2=vertical, 3=rotate90)')
    parser.add_argument('--batch_size', type=int, default=84, help='Batch size for inference')
    parser.add_argument('--crop', type=int, nargs=4, help='Crop coordinates (x1 y1 x2 y2)')
    parser.add_argument('--no_weight_mask', action='store_true', help='Disable weight masking for patches')
    parser.add_argument('--num_workers', type=int, help='Number of worker processes (default: CPU count)')
    parser.add_argument('--patches_in_ram', type=int, default=10000,
                        help='Maximum number of patches to keep in RAM at once (default: 10000)')
    parser.add_argument('--title_crop', action='store_true', help='Use default title crop for specific fragment IDs')
    parser.add_argument('--output_dir', help='Target directory for saving all inference results')

    args = parser.parse_args()

    if not args.end_layer:
        args.end_layer = args.start_layer

    # Validate end_layer is >= start_layer
    if args.end_layer < args.start_layer:
        raise ValueError(f"end_layer ({args.end_layer}) must be >= start_layer ({args.start_layer})")

    # Only title crop or crop allowed but not both
    assert not (args.title_crop and args.crop), "Cannot use both title crop and custom crop coordinates"

    if args.title_crop:
        print("Using default title crop")
        if "02110815/parts_contrasted/02110815_03" in args.fragment_path:
            args.crop = (0, 4500, 4500, 7500)
        elif "03192025/parts_contrasted/03192025_02" in args.fragment_path:
            args.crop = (2700, 4630, 9100, 7730)
        else:
            raise ValueError(f"Title crop requested, but fragment path {args.fragment_path} does not have a title_crop")

    # Run sliding window inference
    print(f"Running sliding window inference on {args.fragment_path}")
    run_sliding_window_inference(
        args.fragment_path,
        args.checkpoint_dir,
        args.start_layer,
        args.end_layer,
        args.stride,
        args.tta_level,
        args.batch_size,
        args.crop,
        not args.no_weight_mask,  # Invert the flag for clearer function parameter
        args.num_workers,
        args.patches_in_ram,
        args.output_dir
    )


if __name__ == "__main__":
    # Set higher sharing priority for subprocesses
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
