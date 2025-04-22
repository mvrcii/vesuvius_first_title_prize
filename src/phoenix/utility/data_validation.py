import glob
import os
import re


def validate_fragments(config, fragments):
    frag_id_2_layers = {}

    for frag_id in fragments:
        val_errors, frag_layers = validate_fragment_files(frag_id=frag_id, cfg=config)
        if len(val_errors) > 0:
            print("Excluded fragment", frag_id)
            print("\n".join(val_errors))
        elif len(frag_layers) > 0:
            print("Fragment", frag_id, "is valid")
            frag_id_2_layers[frag_id] = frag_layers

    return frag_id_2_layers


def find_mask_files(fragment_dir):
    mask_patterns = [
        os.path.join(fragment_dir, "mask.png"),  # Standard mask
        os.path.join(fragment_dir, "*_flat_mask.png"),  # Timestamp masks
        os.path.join(fragment_dir, "*mask.png")  # Timestamp masks
    ]

    found_masks = []
    for pattern in mask_patterns:
        found_masks.extend(glob.glob(pattern))

    return found_masks


def validate_fragment_files(frag_id, cfg):
    errors = []
    frag_dir = os.path.join(f"data/scroll{cfg.scroll_id}/fragments", f"{frag_id}")
    frag_layer_dir = os.path.join(frag_dir, 'layers')

    # Check if fragment directory exists
    if not os.path.isdir(frag_dir):
        errors.append(f"\033[91mReason:\t\tFragment directory '{frag_dir}' does not exist\033[0m")

    # Check if layer directory exists
    if not os.path.isdir(frag_layer_dir):
        errors.append(f"\033[91mReason:\t\tLayer directory {frag_layer_dir} not found in:\n{frag_layer_dir}\033[0m")

    # Check if any mask file exists using the helper function
    mask_files = find_mask_files(frag_dir)
    if not mask_files:
        errors.append(f"\033[91mReason:\t\tNo mask file found in directory:\n{frag_dir}\033[0m")

    # Stop if any errors occurred
    if len(errors) > 0:
        return errors, []

    # Get required 12 channels for this fragment
    if "parts" in frag_id:
        required_channels_start = 5
        required_channels_end = 20
    else:
        raise NotImplementedError("Currently only large chunked auto-segmentations in 21 layer format are supported")

    required_channels = set(range(required_channels_start, required_channels_end + 1))

    # Check for both JPG and TIF formats for layer files
    tif_channels = set(extract_indices(frag_layer_dir, pattern=r'(\d+)\.tif'))
    jpg_channels = set(extract_indices(frag_layer_dir, pattern=r'(\d+)\.jpg'))

    # Combine all found channels
    existing_layer_channels = tif_channels.union(jpg_channels)

    # Check if any required channels are missing
    missing_layer_channels = required_channels - existing_layer_channels
    if missing_layer_channels:
        errors.append(
            f"\033[91mReason:\t\tLayer channel files {format_ranges(sorted(list(missing_layer_channels)))} not found\033[0m")

    # Return only the valid channels that are required
    valid_channels = existing_layer_channels.intersection(required_channels)

    return errors, sorted(list(valid_channels))


def find_consecutive_ch_blocks_of_size(channels, ch_block_size):
    channels = sorted(channels)
    result = []
    i = 0
    while i <= len(channels) - ch_block_size:
        if all(channels[i + j] + 1 == channels[i + j + 1] for j in range(ch_block_size - 1)):
            result.extend(channels[i:i + ch_block_size])
            i += ch_block_size  # Skip to the element after the current block
        else:
            i += 1
    return set(result)


def extract_indices(directory, pattern):
    indices = []

    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            groups = match.groups()
            if len(groups) == 1:
                # Single number (e.g., \d+.tif)
                number = int(groups[0])
                indices.append(number)
            elif len(groups) == 2:
                # Range of numbers (e.g., inklabels_(\d+)_(\d+).png)
                start_layer, end_layer = map(int, groups)
                indices.extend(range(start_layer, end_layer + 1))

    indices = list(set(indices))  # Remove duplicates if any
    indices.sort()  # Sort the indices in ascending order
    return indices


def format_ranges(numbers, file_ending=".tif", digits=5):
    """Convert a list of numbers into a string of ranges."""
    if not numbers:
        return ""

    ranges = []
    start = end = numbers[0]

    for n in numbers[1:]:
        if n - 1 == end:  # Part of the range
            end = n
        else:  # New range
            ranges.append((start, end))
            start = end = n
    ranges.append((start, end))

    return ', '.join(
        [f"{s:0{digits}d}{file_ending} - {e:0{digits}d}{file_ending}" if s != e else f"{s:0{digits}d}.tif" for s, e in
         ranges])
