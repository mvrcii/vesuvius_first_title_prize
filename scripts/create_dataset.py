import argparse
import gc
import glob
import logging
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from multiprocessing import freeze_support

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from tqdm import tqdm

from phoenix.utility.configs import Config
from phoenix.utility.data_validation import validate_fragments
from phoenix.utility.utils import write_to_config, get_frag_name_from_id

Image.MAX_IMAGE_PIXELS = None

# Declare variables that will be initialized in main
LABEL_INFO_LIST = None
GLOBAL_PBAR = None


def chunk_list(input_list, num_chunks):
    """Split a list into evenly sized chunks"""
    chunk_size = math.ceil(len(input_list) / num_chunks)
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def process_fragments_in_chunks(config: Config, frags, only_labels, label_info_list, num_chunks=4):
    """
    Process fragments in sequential chunks to reduce memory usage
    """
    logging.info(f"Processing fragments in {num_chunks} sequential chunks")

    # Validate all fragments upfront to get the complete mapping
    full_frag_id_2_channel = validate_fragments(config, frags)

    # Split the fragments into chunks
    fragment_ids = list(full_frag_id_2_channel.keys())
    chunked_fragment_ids = chunk_list(fragment_ids, num_chunks)

    # Process each chunk sequentially
    for chunk_idx, chunk_frags in enumerate(chunked_fragment_ids):
        logging.info(f"Processing chunk {chunk_idx + 1}/{num_chunks} with {len(chunk_frags)} fragments")

        # Create subset of frag_id_2_channel for this chunk
        chunk_frag_id_2_channel = {frag_id: full_frag_id_2_channel[frag_id] for frag_id in chunk_frags}

        # Process this chunk
        process_fragment_chunk(config, chunk_frag_id_2_channel, only_labels, label_info_list)

        # Force garbage collection after each chunk
        gc.collect()

    # Save the complete label info list to a CSV file
    root_dir = os.path.join(config.dataset_target_dir)
    df = pd.DataFrame(list(label_info_list), columns=['filename', 'frag_id', 'channels', 'ink_p', 'ignore_p'])
    os.makedirs(root_dir, exist_ok=True)
    df.to_csv(os.path.join(root_dir, "label_infos.csv"))

    write_to_config(os.path.join(root_dir),
                    patch_size=config.patch_size,
                    label_size=config.label_size,
                    stride=config.stride,
                    in_chans=config.in_chans,
                    fragment_names=[get_frag_name_from_id(frag_id).upper() for frag_id in
                                    full_frag_id_2_channel.keys()],
                    frag_id_2_channel=full_frag_id_2_channel)


def process_fragment_chunk(config: Config, frag_id_2_channel, only_labels, label_info_list):
    """
    Extract patches from a chunk of fragments in parallel
    """
    logging.info(f"Starting to extract image and label patches for chunk with {len(frag_id_2_channel)} fragments..")

    # Set up global progress bar
    total_fragments = len(frag_id_2_channel)
    global GLOBAL_PBAR
    GLOBAL_PBAR = tqdm(total=total_fragments, desc="Processing fragment chunk")

    # Create a partial function with fixed arguments
    process_fragment_partial = partial(
        process_fragment_parallel,
        config=config,
        only_labels=only_labels,
        label_info_list=label_info_list
    )

    # Calculate optimal number of processes
    num_processes = min(os.cpu_count(), len(frag_id_2_channel))

    # Process fragments in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                process_fragment_partial,
                fragment_id=fragment_id,
                channels=channels
            )
            for fragment_id, channels in frag_id_2_channel.items()
        ]

        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
                GLOBAL_PBAR.update(1)
            except Exception as e:
                logging.error(f"Error processing fragment: {e}")

    # Close progress bar
    GLOBAL_PBAR.close()


def extract_patches(config: Config, frags, only_labels, label_info_list):
    """
    Extract patches from multiple fragments in parallel
    """
    frag_id_2_channel = validate_fragments(config, frags)
    logging.info(f"Starting to extract image and label patches in parallel..")

    # Set up global progress bar
    total_fragments = len(frag_id_2_channel)
    global GLOBAL_PBAR
    GLOBAL_PBAR = tqdm(total=total_fragments, desc="Processing fragments")

    # Create a partial function with fixed arguments
    process_fragment_partial = partial(
        process_fragment_parallel,
        config=config,
        only_labels=only_labels,
        label_info_list=label_info_list
    )

    if len(frag_id_2_channel) == 0:
        raise ValueError("No valid fragments found for processing.")

    # Calculate optimal number of processes
    num_processes = min(os.cpu_count(), len(frag_id_2_channel))

    # Process fragments in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(
                process_fragment_partial,
                fragment_id=fragment_id,
                channels=channels
            )
            for fragment_id, channels in frag_id_2_channel.items()
        ]

        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
                GLOBAL_PBAR.update(1)
            except Exception as e:
                logging.error(f"Error processing fragment: {e}")

    # Close progress bar
    GLOBAL_PBAR.close()

    # Save the label info list to a CSV file
    root_dir = os.path.join(config.dataset_target_dir)
    df = pd.DataFrame(list(label_info_list), columns=['filename', 'frag_id', 'channels', 'ink_p', 'ignore_p'])
    os.makedirs(root_dir, exist_ok=True)
    df.to_csv(os.path.join(root_dir, "label_infos.csv"))

    write_to_config(os.path.join(root_dir),
                    patch_size=config.patch_size,
                    label_size=config.label_size,
                    stride=config.stride,
                    in_chans=config.in_chans,
                    fragment_names=[get_frag_name_from_id(frag_id).upper() for frag_id in frag_id_2_channel.keys()],
                    frag_id_2_channel=frag_id_2_channel)


def process_fragment_parallel(config: Config, fragment_id, channels, only_labels, label_info_list):
    """Wrapper function for parallel processing of fragments"""
    try:
        frag_name = '_'.join([get_frag_name_from_id(fragment_id)]).upper()
        target_dir = os.path.join(config.dataset_target_dir, frag_name)

        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        write_to_config(target_dir, frag_id=fragment_id, channels=channels)

        # Process the fragment
        create_dataset_parallel(target_dir, config, fragment_id, channels, only_labels, label_info_list)

        return True
    except Exception as e:
        logging.error(f"Error processing fragment {fragment_id}: {str(e)}")
        raise e


def clear_dataset(config: Config):
    """Clear existing dataset directory"""
    print("Clearing datasets with patch size", config.patch_size)
    root_dir = os.path.join(config.dataset_target_dir)

    if os.path.isdir(root_dir):
        path = os.path.normpath(root_dir)

        if os.name == 'nt':  # Windows
            import subprocess
            subprocess.run(f'rd /s /q "{path}"', shell=True)
        else:
            import subprocess
            subprocess.run(['rm', '-rf', path])

        print("Deleted existing dataset directory:", root_dir)


def create_dataset_parallel(target_dir, config: Config, frag_id, channels, only_labels, label_info_list):
    """Create dataset with parallelized patch processing"""
    os.makedirs(target_dir, exist_ok=True)

    # determine if contrasted or not and select fragment dir accordingly
    fragment_type = "fragments"

    if getattr(config, 'contrasted', False):
        fragment_type = "fragments_contrasted"

    if "parts" in frag_id:
        fragment_type = "fragments"
    fragment_dir = os.path.join(config.data_root_dir, f"scroll{config.scroll_id}", fragment_type, f"{frag_id}")

    # check for existence of fragment dir
    if not os.path.isdir(fragment_dir):
        raise ValueError(f"Fragment dir does not exist: {fragment_dir}")

    # check for existence of label path => if not in contrasted, check base
    label_path = os.path.join(fragment_dir, f"label.png")
    if not os.path.isfile(label_path):
        label_path = label_path.replace("fragments_contrasted", "fragments")
    assert os.path.isfile(label_path), f"Label file does not exist: {label_path}"

    # check for existence of ignore path => if not in contrasted, check base
    ignore_path = os.path.join(fragment_dir, f"ignore.png")
    if not os.path.isfile(ignore_path):
        ignore_path = ignore_path.replace("fragments_contrasted", "fragments")
    if not os.path.isfile(ignore_path):
        print("WARNING: proceeding WITHOUT ignore mask for this fragment")

    # Load mask
    if "parts" in frag_id:
        standard_mask = os.path.join(fragment_dir, f"mask.png")
    else:
        standard_mask = os.path.join(fragment_dir.replace("_contrasted", ""), "mask.png")
        standard_mask = standard_mask.replace("_contrasted", "")
        timestamp_masks = glob.glob(os.path.join(fragment_dir.replace("_contrasted", ""), "*mask.png"))

    if os.path.isfile(standard_mask):
        mask_path = standard_mask
    elif timestamp_masks:
        timestamp_mask = timestamp_masks[0].replace("_contrasted", "")
        mask_path = timestamp_mask
    else:
        raise ValueError(f"No mask file found for fragment: {frag_id}")

    # Load mask in main process
    mask = load_binary_image(mask_path, config.patch_size)

    start_channel, end_channel = min(channels), max(channels)
    read_chans = range(start_channel, end_channel + 1)

    # Only load image data if we're not only processing labels
    if not only_labels:
        image_tensor = read_fragment_images_for_channels(
            fragment_dir,
            config.patch_size,
            channels=read_chans,
            ch_block_size=config.in_chans
        )
    else:
        image_tensor = None

    # Load label array
    label_arr = load_binary_image(image_path=label_path, patch_size=config.patch_size)

    # load ignore array
    if not os.path.isfile(ignore_path):
        ignore_arr = np.zeros_like(label_arr)
    else:
        ignore_arr = load_binary_image(image_path=ignore_path, patch_size=config.patch_size)

    if label_arr.sum() == 0:
        print("Warning: Label array is empty")

    if only_labels:
        assert label_arr.shape == mask.shape == ignore_arr.shape, (
            f"Shape mismatch for Fragment {frag_id}:"
            f"Mask={mask.shape} Label={label_arr.shape}")
    else:
        assert label_arr.shape == mask.shape == image_tensor[0].shape == ignore_arr.shape, (
            f"Shape mismatch for Fragment {frag_id}: Img={image_tensor[0].shape} "
            f"Mask={mask.shape} Label={label_arr.shape}")

    # Process patches in parallel
    results = process_channel_stack_parallel(
        config=config,
        target_dir=target_dir,
        frag_id=frag_id,
        mask=mask,
        img_tensor=image_tensor,
        label_arr=label_arr,
        ignore_arr=ignore_arr,
        start_channel=start_channel,
        only_labels=only_labels,
        label_info_list=label_info_list
    )

    # Clean up memory
    del image_tensor, label_arr
    gc.collect()

    patch_cnt, skipped_cnt, ignore_skipped_count = results
    total_patches = patch_cnt + skipped_cnt + ignore_skipped_count
    print(
        f"Fragment {frag_id}: Total={total_patches} | Patches={patch_cnt} | Skipped={skipped_cnt} | Ignored={ignore_skipped_count}")


def process_channel_stack_parallel(config: Config, target_dir, frag_id, mask, img_tensor, label_arr, ignore_arr,
                                   start_channel, only_labels, label_info_list):
    """Process channel stack with parallel patch extraction"""
    # Create patch coordinates
    x1_list = list(range(0, label_arr.shape[1] - config.patch_size + 1, config.stride))
    y1_list = list(range(0, label_arr.shape[0] - config.patch_size + 1, config.stride))

    # Create coordinate pairs for all patches
    coordinates = [(y1, x1) for y1 in y1_list for x1 in x1_list]

    # Create output directories
    img_dest_dir = os.path.join(target_dir, "images")
    label_dest_dir = os.path.join(target_dir, "labels")
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(label_dest_dir, exist_ok=True)

    # Validate inputs
    if label_arr.ndim != 2:
        raise ValueError(f"Invalid label arr shape: {label_arr.shape}")

    if not only_labels:
        # Image data validation
        if img_tensor.ndim != 3 or img_tensor.shape[0] != config.in_chans or len(img_tensor[0]) + len(
                img_tensor[1]) == 0:
            raise ValueError(f"Expected tensor with shape ({config.in_chans}, height, width), got {img_tensor.shape}")

    # Create partial function with fixed arguments
    process_patch_partial = partial(
        process_single_patch,
        config=config,
        frag_id=frag_id,
        mask=mask,
        img_tensor=img_tensor,
        label_arr=label_arr,
        ignore_arr=ignore_arr,
        start_channel=start_channel,
        only_labels=only_labels,
        img_dest_dir=img_dest_dir,
        label_dest_dir=label_dest_dir
    )

    # Determine optimal number of threads
    num_threads = min(os.cpu_count() * 2, len(coordinates))

    # Create counters for statistics
    patch_counter = mp.Value('i', 0)
    mask_skipped_counter = mp.Value('i', 0)
    ignore_skipped_counter = mp.Value('i', 0)

    # Process patches in parallel using thread pool
    local_results = []
    chunk_size = max(1, len(coordinates) // (num_threads * 4))  # Process in chunks for better performance

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks in batches
        for i in range(0, len(coordinates), chunk_size):
            batch = coordinates[i:i + chunk_size]
            futures = [executor.submit(process_patch_partial, y1=y1, x1=x1,
                                       patch_counter=patch_counter,
                                       mask_skipped_counter=mask_skipped_counter,
                                       ignore_skipped_counter=ignore_skipped_counter)
                       for y1, x1 in batch]

            # Collect results from this batch
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        local_results.append(result)
                except Exception as e:
                    logging.error(f"Error processing patch: {e}")

    # Add results to shared list
    label_info_list.extend(local_results)

    # Return statistics
    return patch_counter.value, mask_skipped_counter.value, ignore_skipped_counter.value


def process_single_patch(config, frag_id, mask, img_tensor, label_arr, ignore_arr, start_channel, only_labels,
                         img_dest_dir, label_dest_dir, y1, x1, patch_counter, mask_skipped_counter,
                         ignore_skipped_counter):
    """Process a single patch and save it to disk if valid"""
    try:
        y2 = y1 + config.patch_size
        x2 = x1 + config.patch_size
        label_shape = (config.label_size, config.label_size)

        # Get mask patch
        mask_patch = mask[y1:y2, x1:x2]

        # Check if patch is fully in mask
        if mask_patch.all() != 1:
            with mask_skipped_counter.get_lock():
                mask_skipped_counter.value += 1
            return None

        # Check if mask patch shape is valid
        if mask_patch.shape != (config.patch_size, config.patch_size):
            with mask_skipped_counter.get_lock():
                mask_skipped_counter.value += 1
            return None

        # Get label and create ignore patch
        label_patch = label_arr[y1:y2, x1:x2]
        ignore_patch = ignore_arr[y1:y2, x1:x2]

        # Set label patch to 0 where ignore patch is 1
        label_patch[ignore_patch == 1] = 0

        # Create keep_patch by inverting ignore patch
        keep_patch = np.logical_not(ignore_patch)

        # Check shapes
        assert label_patch.shape == (config.patch_size,
                                     config.patch_size), f"Label patch wrong shape: {label_patch.shape}"

        # Scale label and keep_patch patch down to label size
        label_patch = resize(label_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)
        keep_patch = resize(keep_patch, label_shape, order=0, preserve_range=True, anti_aliasing=False)

        # Calculate percentages
        label_pixel_count = np.prod(label_patch.shape)
        ink_percentage = int((label_patch.sum() / label_pixel_count) * 100)
        keep_percent = int((keep_patch.sum() / np.prod(keep_patch.shape)) * 100)
        ignore_percent = 100 - keep_percent

        # Discard images with less than 5% keep pixels
        if keep_percent < 5:
            with ignore_skipped_counter.get_lock():
                ignore_skipped_counter.value += 1
            return None

        # Create file name
        file_name = f"f{frag_id.replace('/', '_')}_ch{start_channel:02d}_{x1}_{y1}_{x2}_{y2}.npy"

        # Save image if needed
        if not only_labels:
            image_patch = img_tensor[:, y1:y2, x1:x2]
            assert image_patch.shape == (config.in_chans, config.patch_size, config.patch_size), \
                f"Image patch wrong shape: {image_patch.shape}"
            np.save(os.path.join(img_dest_dir, file_name), image_patch)

        # Stack label and keep patch, then save
        label_patch = np.stack([label_patch, keep_patch], axis=0)
        label_patch = np.packbits(label_patch.flatten())
        np.save(os.path.join(label_dest_dir, file_name), label_patch)

        # Increment patch counter
        with patch_counter.get_lock():
            patch_counter.value += 1

        # Return information for this patch
        return (file_name, frag_id, start_channel, ink_percentage, ignore_percent)

    except Exception as e:
        logging.error(f"Error processing patch at ({y1}, {x1}): {str(e)}")
        return None


def pad_to_patch_size(image: np.ndarray, patch_size: int):
    """
    Returns the image, padded such that it is evenly divisible by patch size
    """
    pad0 = (patch_size - image.shape[0] % patch_size) % patch_size
    pad1 = (patch_size - image.shape[1] % patch_size) % patch_size
    return np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)


def read_fragment_images_for_channels(root_dir, patch_size, channels, ch_block_size):
    """Read fragment images for channels with improved memory management"""
    images = []

    for channel in channels:
        try:
            img_path = os.path.join(root_dir, "layers", f"{channel:02}.jpg")
            assert os.path.isfile(img_path), "Fragment file does not exist: " + img_path
        except:
            img_path = os.path.join(root_dir, "layers", f"{channel:02}.tif")

        assert os.path.isfile(img_path), "Fragment file does not exist: " + img_path

        # Use IMREAD_UNCHANGED to preserve original bit depth (16-bit TIFFs)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Normalize 16-bit images to 8-bit range if needed
        if image.dtype == np.uint16:
            print(f"Converting 16-bit image to 8-bit: {img_path}")
            # Scale to 8-bit (0-255) range while preserving relative values
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image is None or image.shape[0] == 0:
            print("Image is empty or not loaded correctly:", img_path)

        # Check image valid range - allow both 8-bit and 16-bit images
        img_max = np.asarray(image).max()
        if not (1 < img_max <= 65535):
            raise ValueError(f"Invalid image pixel range, max value: {img_max}")

        image = pad_to_patch_size(image, patch_size)
        images.append(image)

    # Stack images along first dimension
    images = np.stack(images, axis=0)
    assert images.ndim == 3 and images.shape[0] == ch_block_size, \
        f"Images shape {images.shape}, ch_block_size {ch_block_size}"

    return np.array(images)


def load_binary_image(image_path, patch_size):
    """
    Loads image from path (mask / label) and returns it as numpy array, padded to be divisible by patch size,
    ensuring it only contains 1's and 0's
    """
    label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert label is not None and label.shape[0] != 0 and label.shape[1] != 0, \
        "Label is empty or not loaded correctly" + str(label.shape)
    label = pad_to_patch_size(label, patch_size)
    label = (label / 255).astype(np.uint8)
    assert set(np.unique(np.array(label))).issubset({0, 1}), "Invalid label"
    return label


if __name__ == '__main__':
    # This is required for multiprocessing on Windows
    freeze_support()

    # Set up multiprocessing start method
    if os.name != 'nt':  # Not needed on Windows as it always uses 'spawn'
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Create a dataset for segmentation based on a given config')
    parser.add_argument('config_path', type=str)
    parser.add_argument(
        '--ide_is_closed',
        action='store_true',
        help="Confirm that this is run from outside an IDE to prevent crashes due to massive file changes."
    )
    parser.add_argument(
        '--only_labels',
        action='store_true',
        help="Don't recreate image patches"
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help="Number of processes to use (default: number of CPU cores)"
    )
    parser.add_argument(
        '--threads_per_process',
        type=int,
        default=None,
        help="Number of threads per process (default: 2x number of CPU cores)"
    )
    parser.add_argument(
        '--memory_limit',
        type=int,
        default=None,
        help="Memory limit in GB (default: 80% of available RAM)"
    )
    parser.add_argument(
        '--chunks',
        type=int,
        default=1,
        help="Number of chunks to split fragments into for sequential processing (default: 1, meaning no chunking)"
    )

    args = parser.parse_args()

    if not args.ide_is_closed:
        error_msg = "Please run this script from outside an IDE to prevent crashes due to massive file changes, and confirm this by setting the --ide_is_closed flag."
        print(f"\033[91m{error_msg}\033[0m")
        exit(1)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    cfg = Config.load_from_file(args.config_path)

    # Set number of processes and threads if provided
    if args.processes:
        os.environ['PYTHONEXECUTOR_PROCESSES'] = str(args.processes)
    if args.threads_per_process:
        os.environ['PYTHONEXECUTOR_THREADS_PER_PROCESS'] = str(args.threads_per_process)
    if args.memory_limit:
        os.environ['PYTHONEXECUTOR_MEMORY_LIMIT_GB'] = str(args.memory_limit)

    # Initialize the manager and shared list here (not at module level)
    manager = mp.Manager()
    label_info_list = manager.list()

    # Clear the dataset if not only processing labels
    if not args.only_labels:
        clear_dataset(config=cfg)

    # Get fragments and extract patches
    fragments = list(set(cfg.fragment_ids).union(cfg.validation_fragments))

    # Measure execution time
    start_time = time.time()

    # Use chunking if specified
    if args.chunks > 1:
        process_fragments_in_chunks(cfg, fragments, only_labels=args.only_labels,
                                    label_info_list=label_info_list, num_chunks=args.chunks)
    else:
        # Use the original extract_patches function if no chunking is requested
        extract_patches(cfg, fragments, only_labels=args.only_labels, label_info_list=label_info_list)

    end_time = time.time()

    print(f"Total execution time: {end_time - start_time:.2f} seconds")
