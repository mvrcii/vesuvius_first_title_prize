import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from phoenix.utility.dataloader import create_data_loader

logger = logging.getLogger(__name__)


# Module-level function for multiprocessing
def _validate_single_tiff(file_path):
    """Validate a single TIFF file - must be outside class for multiprocessing"""
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return file_path, False

    try:
        with tifffile.TiffFile(file_path) as tif:
            # Try to access basic metadata without loading full image
            if len(tif.pages) == 0:
                return file_path, False
            _ = tif.pages[0].shape
            return file_path, True
    except Exception:
        return file_path, False


# Module-level function for parallel TIFF writing
def _write_layer_to_tiff(args):
    """Write a layer to a TIFF file"""
    layer_np, output_file = args
    try:
        # Create temporary file first, then rename to final path when complete
        temp_output_file = output_file + ".tmp"
        tifffile.imwrite(temp_output_file, layer_np)

        # Only rename to final file if write was successful
        os.replace(temp_output_file, output_file)
        return output_file, True
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_output_file):
            os.remove(temp_output_file)
        return output_file, (False, str(e))


def _create_mask(args):
    """Create mask from a 10.tif file - must be outside class for multiprocessing"""
    tif_path, output_path = args
    try:
        # Open the TIF file
        img = Image.open(tif_path)

        # Convert to numpy array for processing
        img_array = np.array(img)

        # Create mask: True where all RGB values are 0 (pure black), False elsewhere
        if len(img_array.shape) == 3:  # RGB image
            mask = np.all(img_array == 0, axis=2)
        else:  # Grayscale image
            mask = (img_array == 0)

        # Convert to image format: 0 for black, 255 for white
        # Invert mask since we want black to stay black, everything else white
        mask_img = Image.fromarray((~mask).astype(np.uint8) * 255)

        # Save as PNG
        mask_img.save(output_path)
        return output_path, True
    except Exception as e:
        return output_path, (False, str(e))


class FragmentSplitter:
    """Class for splitting a large fragment into smaller pieces along the horizontal axis."""

    def _validate_tiff_parallel(self, file_paths, num_processes=None):
        """
        Validate multiple TIFF files in parallel using multiprocessing.

        Args:
            file_paths: List of file paths to validate
            num_processes: Number of processes to use (default: CPU count)

        Returns:
            dict: Dictionary mapping file paths to validation results (True/False)
        """
        import multiprocessing as mp

        # Fall back to sequential validation for small batches or if multiprocessing fails
        if len(file_paths) < 4 or num_processes == 1:
            return {file_path: self._validate_tiff_sequential(file_path) for file_path in file_paths}

        try:
            if num_processes is None:
                num_processes = mp.cpu_count()

            # Create a pool and map the validation function to all files
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(_validate_single_tiff, file_paths)

            # Convert results list to dictionary
            return dict(results)
        except Exception as e:
            logger.warning(f"Multiprocessing failed: {str(e)}. Falling back to sequential validation.")
            return {file_path: self._validate_tiff_sequential(file_path) for file_path in file_paths}

    def _validate_tiff_sequential(self, file_path):
        """Validate a single TIFF file using sequential processing"""
        if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
            return False

        try:
            with tifffile.TiffFile(file_path) as tif:
                if len(tif.pages) == 0:
                    return False
                # Try to access basic metadata without loading the entire image
                _ = tif.pages[0].shape
                return True
        except Exception:
            return False

    def __init__(self, fragment_id, scroll_id=5, base_path="data", contrasted=False, verbose=False):
        """
        Initialize the fragment splitter and ensure zarr store exists.

        Args:
            fragment_id: ID of the fragment to split
            scroll_id: ID of the scroll (default: 5)
            base_path: Base path to the data directory
            verbose: Whether to print verbose output
        """
        self.fragment_id = fragment_id
        self.scroll_id = scroll_id
        self.base_path = base_path
        self.fragments_path = os.path.join(base_path, f"scroll{scroll_id}", "fragments")
        self.fragment_path = os.path.join(self.fragments_path, fragment_id)
        self.layers_dir = os.path.join(self.fragment_path, "layers")
        self.verbose = verbose
        self.contrasted = contrasted

        if self.verbose:
            logger.info(f"Ensuring zarr store exists for scroll {scroll_id}, fragment {fragment_id}")

        # Create zarr data loader for the source fragment (this creates the store if needed)
        self.data_loader = create_data_loader(
            fragment_path=self.fragment_path,
            use_zarr=False,
            verbose=verbose,
            contrasted=contrasted
        )

        self.height = self.data_loader.height
        self.width = self.data_loader.width

        if hasattr(self.data_loader, 'min_layer_idx') and hasattr(self.data_loader, 'max_layer_idx'):
            self.min_layer_idx = self.data_loader.min_layer_idx
            self.max_layer_idx = self.data_loader.max_layer_idx
            self.num_layers = self.max_layer_idx - self.min_layer_idx + 1
        else:
            # Count files in layers directory
            layer_files = [f for f in os.listdir(self.layers_dir) if os.path.isfile(os.path.join(self.layers_dir, f))]
            self.num_layers = len(layer_files)
            self.min_layer_idx = 0
            self.max_layer_idx = self.num_layers - 1

        if verbose:
            logger.info(f"Source fragment dimensions: {self.width}x{self.height}")
            logger.info(f"Number of layers: {self.num_layers} (index {self.min_layer_idx} to {self.max_layer_idx})")

    def split_fragment(self, part_width=10000, num_processes=None, specific_chunks=None, first_chunk_offset=0):
        """
        Split the fragment into parts with specified width.

        - Parallel file validation
        - Loads one layer at a time but writes to disk in parallel

        Args:
            part_width: Width of each part in pixels
            num_processes: Number of processes for parallel validation (default: CPU count)
            specific_chunks: List of specific chunk indices to process (1-indexed)
            first_chunk_offset: Offset in pixels for the first chunk (default: 0)
                           If > 0, the first chunk will be (part_width - first_chunk_offset) pixels wide

        Returns:
            List of paths to created fragment parts
        """
        import multiprocessing as mp
        num_processes = mp.cpu_count() if num_processes is None else num_processes

        # Validate first_chunk_offset
        if first_chunk_offset < 0 or first_chunk_offset >= part_width:
            raise ValueError(f"first_chunk_offset must be between 0 and {part_width - 1}")

        # Calculate chunk boundaries
        chunk_boundaries = []

        # First chunk handling
        first_chunk_width = part_width - first_chunk_offset if first_chunk_offset > 0 else part_width
        if first_chunk_width >= self.width:
            # Only one chunk needed
            chunk_boundaries = [(0, self.width)]
        else:
            # Start with the first chunk
            chunk_boundaries.append((0, first_chunk_width))

            # Add remaining full chunks
            start_x = first_chunk_width
            while start_x < self.width:
                end_x = min(start_x + part_width, self.width)
                chunk_boundaries.append((start_x, end_x))
                start_x = end_x

        # Number of parts
        num_parts = len(chunk_boundaries)

        # Calculate number of parts
        if self.verbose:
            if first_chunk_offset > 0:
                logger.info(f"First chunk has offset {first_chunk_offset}, width: {first_chunk_width}")
            logger.info(f"Splitting into {num_parts} parts")
            for i, (start_x, end_x) in enumerate(chunk_boundaries):
                logger.info(f"  Part {i + 1}: start={start_x}, end={end_x}, width={end_x - start_x}")

        created_paths = []

        # Handle specific chunks (convert to 0-indexed internally)
        if specific_chunks:
            parts_to_process = [x - 1 for x in specific_chunks]
            # Ensure we only process valid parts
            parts_to_process = [p for p in parts_to_process if 0 <= p < num_parts]

            if not parts_to_process:
                raise ValueError("No valid parts to process. Check the specific_chunks argument.")
        else:
            parts_to_process = list(range(num_parts))

        part_iterator = tqdm(parts_to_process, desc="Processing fragment parts", unit="part")

        # Create a process pool executor for parallel TIFF writing
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for part_idx in part_iterator:
                try:
                    # Get boundaries for this part
                    start_x, end_x = chunk_boundaries[part_idx]
                    actual_width = end_x - start_x

                    # Create directory for this part within original fragment parts folder
                    part_id = f"{self.fragment_id}_{part_idx + 1:02d}"
                    contrasted_str = "_contrasted" if self.contrasted else ""
                    parts_dir = os.path.join(self.fragment_path, f"parts{contrasted_str}")
                    os.makedirs(parts_dir, exist_ok=True)
                    part_path = os.path.join(parts_dir, part_id)
                    layers_path = os.path.join(part_path, "layers")

                    part_iterator.set_description(f"Part {part_idx + 1}/{num_parts}: {part_id}")

                    # Validate existing files in parallel
                    if os.path.exists(layers_path):
                        expected_files = [
                            f"{str(i).zfill(self.data_loader.padding)}.{self.data_loader.file_type}"
                            for i in range(self.min_layer_idx, self.max_layer_idx + 1)
                        ]
                        file_paths = [os.path.join(layers_path, f) for f in expected_files]
                        part_iterator.set_postfix(status="validating", files=len(file_paths))
                        validation_results = self._validate_tiff_parallel(file_paths, num_processes)

                        # Check validation results
                        all_files_valid = all(validation_results.values())
                        corrupt_count = sum(1 for valid in validation_results.values() if not valid)

                        if not all_files_valid and corrupt_count > 0:
                            part_iterator.set_postfix(status="corrupted", files=corrupt_count)

                        if all_files_valid:
                            part_iterator.set_postfix(status="complete")
                            created_paths.append(part_path)
                            continue  # Skip to next part if all files are valid

                    # Determine which layers need processing
                    if not os.path.exists(layers_path):
                        # New part - process all layers
                        os.makedirs(layers_path, exist_ok=True)
                        layers_to_process = list(range(self.min_layer_idx, self.max_layer_idx + 1))
                        part_iterator.set_postfix(status="new part")
                    else:
                        # Check existing files in parallel
                        file_paths = []
                        for layer_idx in range(self.min_layer_idx, self.max_layer_idx + 1):
                            filename = f"{str(layer_idx).zfill(self.data_loader.padding)}.{self.data_loader.file_type}"
                            output_file = os.path.join(layers_path, filename)
                            file_paths.append(output_file)

                        # Validate all files in parallel
                        part_iterator.set_postfix(status="checking files")
                        validation_results = self._validate_tiff_parallel(file_paths, num_processes)

                        # Create list of layers to process based on validation results
                        layers_to_process = []
                        for layer_idx in range(self.min_layer_idx, self.max_layer_idx + 1):
                            filename = f"{str(layer_idx).zfill(self.data_loader.padding)}.{self.data_loader.file_type}"
                            output_file = os.path.join(layers_path, filename)

                            if not validation_results.get(output_file, False):
                                layers_to_process.append(layer_idx)

                    if not layers_to_process:
                        part_iterator.set_postfix(status="complete")
                        created_paths.append(part_path)
                        continue  # Skip to next part if nothing to process

                    # We have layers to process
                    part_iterator.set_postfix(status=f"processing {len(layers_to_process)} layers")

                    # Process each layer - chunk loading one at a time
                    # but use parallelism for writing to disk
                    futures = []

                    # Use tqdm for layer processing
                    layer_iterator = tqdm(
                        layers_to_process,
                        desc=f"Processing layers for part {part_idx + 1}",
                        leave=False
                    )

                    # We'll use a batch approach to avoid too many parallel tasks
                    batch_size = min(num_processes * 2, len(layers_to_process))
                    batch_futures = []

                    for layer_idx in layer_iterator:
                        # Load single layer for the current part
                        layer_np = self.data_loader.load_chunk(
                            layer_start=layer_idx,
                            num_layers=1,
                            start_x=start_x,
                            end_x=end_x
                        )[0]  # Extract the single layer from the returned array

                        filename = f"{str(layer_idx).zfill(self.data_loader.padding)}.{self.data_loader.file_type}"
                        output_file = os.path.join(layers_path, filename)

                        # Submit to process pool for parallel writing
                        future = executor.submit(_write_layer_to_tiff, (layer_np, output_file))
                        batch_futures.append((output_file, future))

                        # If we've reached the batch size, wait for all tasks to complete
                        if len(batch_futures) >= batch_size:
                            for out_file, fut in batch_futures:
                                futures.append((out_file, fut.result()))
                            batch_futures = []

                    # Process any remaining batch items
                    for out_file, fut in batch_futures:
                        futures.append((out_file, fut.result()))

                    # Check for errors
                    errors = [(f, r[1]) for f, r in futures if not isinstance(r, bool) and not r[0]]
                    if errors:
                        part_iterator.set_postfix(status=f"completed with {len(errors)} errors")
                        for output_file, error in errors:
                            logger.error(f"Error processing {output_file}: {error}")
                    else:
                        part_iterator.set_postfix(status="completed")

                    created_paths.append(part_path)

                except Exception as e:
                    part_iterator.set_postfix(status=f"error: {str(e)[:30]}")
                    logger.error(f"Error processing part {part_idx + 1}: {str(e)}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()

        return created_paths


if __name__ == "__main__":
    def parse_comma_separated_ints(arg):
        return [int(x) for x in arg.split(',')]


    parser = argparse.ArgumentParser(description="Split a large fragment into smaller pieces")
    parser.add_argument("fragment_id", help="ID of the fragment to split")
    parser.add_argument("--scroll-id", type=int, default=5,
                        help="ID of the scroll (default: 5)")
    parser.add_argument("--base-path", default="data",
                        help="Base path to the data directory (default: data)")
    parser.add_argument("--part-width", type=int, default=10000,
                        help="Width of each part (default: 10000)")
    parser.add_argument("--first-chunk-offset", type=int, default=0,
                        help="Offset in pixels for the first chunk. If > 0, first chunk will be (part_width - offset) pixels wide (default: 0)")
    parser.add_argument("--create-zarr", action="store_true",
                        help="Create zarr stores for split fragments")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of processes for parallel validation (default: CPU count)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    parser.add_argument("--contrasted", action="store_true",
                        help="Use contrasted data loader")
    parser.add_argument("--specific-chunks", "-ch", type=parse_comma_separated_ints,
                        help="Specific chunk numbers to process 0-indexed (comma-separated, e.g., 0,3,4,11,24)")
    args = parser.parse_args()

    splitter = FragmentSplitter(
        fragment_id=args.fragment_id,
        scroll_id=args.scroll_id,
        base_path=args.base_path,
        verbose=args.verbose,
        contrasted=args.contrasted
    )

    created_paths = splitter.split_fragment(
        part_width=args.part_width,
        num_processes=args.processes,
        specific_chunks=args.specific_chunks,
        first_chunk_offset=args.first_chunk_offset
    )

    print(f"Created {len(created_paths)} fragment parts:")
    for path in created_paths:
        print(f"  {path}")
