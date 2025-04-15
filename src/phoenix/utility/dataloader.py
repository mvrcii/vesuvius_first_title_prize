import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import dask.array as da
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import tifffile
import zarr
from dask.diagnostics import ProgressBar
from scipy.interpolate import CubicSpline
from tqdm import tqdm

logger = logging.getLogger(__name__)


def create_lut(control_points, bit_depth=8):
    """
    Create a lookup table (LUT) based on control points for the given bit depth.

    Args:
        control_points (list of tuples): Control points in 8-bit range, e.g. [(0,0), (128,64), (255,255)]
        bit_depth (int): Bit depth of the image (8 or 16)

    Returns:
        np.ndarray: The generated LUT as a 1D numpy array.
    """
    if bit_depth == 16:
        # Scale control points from 8-bit to 16-bit (0-255 -> 0-65535)
        control_points = [(x * 257, y * 257) for x, y in control_points]
        x_points, y_points = zip(*control_points)
        x_range = np.linspace(0, 65535, 65536)
        spline = CubicSpline(x_points, y_points)
        lut = spline(x_range)
        lut = np.clip(lut, 0, 65535).astype('uint16')
    elif bit_depth == 8:
        x_points, y_points = zip(*control_points)
        x_range = np.linspace(0, 255, 256)
        spline = CubicSpline(x_points, y_points)
        lut = spline(x_range)
        lut = np.clip(lut, 0, 255).astype('uint8')
    else:
        raise ValueError("Unsupported bit depth: {}".format(bit_depth))
    return lut


def apply_lut(image, lut):
    """
    Apply a lookup table (LUT) to an image.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        lut (np.ndarray): 1D lookup table of appropriate size.

    Returns:
        np.ndarray: Transformed image.
    """
    if len(image.shape) == 2:  # Grayscale
        transformed_image = lut[image]
    elif len(image.shape) == 3:  # Color image
        transformed_image = np.stack([lut[image[:, :, i]] for i in range(image.shape[2])], axis=-1)
    else:
        raise ValueError("Unsupported image format with shape: {}".format(image.shape))
    return transformed_image


class FragmentDataLoader(ABC):
    """Abstract base class for fragment image data loading."""

    def __init__(self, fragment_path: str, verbose: bool = False):
        self.fragment_path = fragment_path
        self.layers_dir = os.path.join(fragment_path, "layers")
        self.verbose = verbose

        first_file = sorted(os.listdir(self.layers_dir))[0]
        file_components = first_file.split(".")
        self.padding = len(file_components[0])
        self.file_type = file_components[-1]

        self.height, self.width = self.get_fragment_dimensions()
        if self.verbose:
            logger.info(f"Fragment dimensions: {self.width}x{self.height}")

    @abstractmethod
    def get_fragment_dimensions(self) -> Tuple[int, int]:
        pass

    def load_chunk(self, layer_start: int, num_layers: int,
                   start_x: int, end_x: int,
                   start_y: int = None, end_y: int = None) -> np.ndarray:
        """Load a chunk from the data with both vertical and horizontal slicing."""
        pass

    def get_layer_path(self, layer_idx: int) -> str:
        return os.path.join(self.layers_dir, f"{str(layer_idx).zfill(self.padding)}.{self.file_type}")

    @staticmethod
    def pad_to_multiple(data: np.ndarray, multiple: int) -> np.ndarray:
        if data.ndim == 2:
            height_pad = -data.shape[0] % multiple
            width_pad = -data.shape[1] % multiple
            if height_pad > 0 or width_pad > 0:
                return np.pad(data, ((0, height_pad), (0, width_pad)), mode='constant')
            return data
        elif data.ndim == 3:
            depth, height, width = data.shape
            height_pad = -height % multiple
            width_pad = -width % multiple
            if height_pad > 0 or width_pad > 0:
                return np.pad(data, ((0, 0), (0, height_pad), (0, width_pad)), mode='constant')
            return data
        raise ValueError(f"Unsupported data dimensions: {data.shape}")

    @staticmethod
    def _process_image_with_lut(path, lut):
        """Process a single image by applying a lookup table.

        This is a standalone method to enable parallel processing.

        Args:
            path: Path to the image file
            lut: Lookup table to apply

        Returns:
            Processed image with LUT applied
        """
        if path.lower().endswith(('.tif', '.tiff')):
            img = tifffile.imread(path)
        else:  # JPG/JPEG
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return apply_lut(img, lut)

    def plot_sample(self,
                    layer_start: Optional[int] = None,
                    num_layers: int = 1,
                    square_size: Optional[int] = None,
                    seed: Optional[int] = None,
                    max_attempts: int = 3,
                    x_range: Optional[Tuple[int, int]] = None,
                    y_range: Optional[Tuple[int, int]] = None,
                    output_path: Optional[str] = None) -> None:
        """
        Load a chunk from the data and plot an ROI that is not completely black.
        The ROI is defined either by a square of given size (with random location) or by explicit
        coordinates if x_range is provided. If y_range is not provided when x_range is given,
        the full y-range is used.

        If the randomly selected ROI is completely black, try again (up to max_attempts).
        In the explicit coordinate case, no retries are attempted.

        Parameters:
          layer_start: Specific layer to inspect; if not provided, one is chosen randomly.
          num_layers: Number of layers to load (default is 1).
          square_size: Size (width and height) of the square ROI (default is 100).
          seed: Optional seed for reproducibility.
          max_attempts: Maximum number of attempts when choosing a random ROI (default is 3).
          x_range: Optional tuple (start_x, end_x) specifying the x coordinates of the ROI.
          y_range: Optional tuple (start_y, end_y) specifying the y coordinates of the ROI.
                   If not provided when x_range is given, the full y-range is used.
          output_path: Optional path to save the resulting ROI as a TIFF file.
        """
        if seed is not None:
            np.random.seed(seed)

        # Choose layer if not provided
        if layer_start is None:
            if hasattr(self, 'min_layer_idx') and hasattr(self, 'max_layer_idx'):
                layer_start = np.random.randint(self.min_layer_idx, self.max_layer_idx + 1)
            else:
                layer_files = sorted(os.listdir(self.layers_dir))
                layer_start = np.random.randint(0, len(layer_files))

        # Explicit ROI if x_range is provided.
        if x_range is not None:
            roi_start_x, roi_end_x = x_range
            # Use full y range if y_range is not provided.
            if y_range is not None:
                roi_start_y, roi_end_y = y_range
            else:
                roi_start_y = 0
                roi_end_y = self.height

            # Validate coordinates.
            if roi_start_x < 0 or roi_start_x >= roi_end_x or roi_end_x > self.width:
                raise ValueError("Provided x_range is out of bounds or invalid.")
            if roi_start_y < 0 or roi_start_y >= roi_end_y or roi_end_y > self.height:
                raise ValueError("Provided y_range is out of bounds or invalid.")

            chunk = self.load_chunk(
                layer_start, num_layers,
                start_y=roi_start_y, end_y=roi_end_y,
                start_x=roi_start_x, end_x=roi_end_x
            )[0]

            if np.all(chunk == 0):
                raise ValueError("The specified ROI is completely black.")

            logger.info(
                f"Selected explicit ROI: layer={layer_start}, "
                f"x=[{roi_start_x}:{roi_end_x}], y=[{roi_start_y}:{roi_end_y}], "
                f"dtype={chunk.dtype}, min={chunk.min()}, max={chunk.max()}"
            )
        else:
            # Use random square ROI.
            if square_size is None:
                square_size = 100

            if self.height < square_size or self.width < square_size:
                raise ValueError("Square size exceeds image dimensions.")

            attempt = 0
            chunk = None
            while attempt < max_attempts:
                roi_start_y = np.random.randint(0, self.height - square_size + 1)
                roi_start_x = np.random.randint(0, self.width - square_size + 1)
                roi_end_y = roi_start_y + square_size
                roi_end_x = roi_start_x + square_size

                chunk = self.load_chunk(
                    layer_start, num_layers,
                    start_y=roi_start_y, end_y=roi_end_y,
                    start_x=roi_start_x, end_x=roi_end_x
                )[0]

                if not np.all(chunk == 0):
                    logger.info(
                        f"Selected ROI: layer={layer_start}, "
                        f"x=[{roi_start_x}:{roi_end_x}], y=[{roi_start_y}:{roi_end_y}], "
                        f"dtype={chunk.dtype}, min={chunk.min()}, max={chunk.max()}"
                    )
                    break
                else:
                    logger.info(f"Attempt {attempt + 1}: ROI was completely black. Retrying...")
                attempt += 1

            if chunk is None or np.all(chunk == 0):
                raise ValueError(f"Could not find a non-black ROI after {max_attempts} attempts.")

        # Save the ROI as a TIFF file if an output path is provided.
        if output_path is not None:
            from PIL import Image
            # Optionally, scale to uint8 if necessary.
            if chunk.dtype != 'uint8':
                chunk_min, chunk_max = chunk.min(), chunk.max()
                if chunk_max > chunk_min:
                    chunk_scaled = 255 * (chunk - chunk_min) / (chunk_max - chunk_min)
                else:
                    chunk_scaled = chunk.copy()
                chunk = chunk_scaled.astype('uint8')
            Image.fromarray(chunk).save(output_path, format='TIFF')
            logger.info(f"ROI saved to {output_path}")

        plt.imshow(chunk, cmap='gray')
        plt.axis('off')
        plt.show()


class ImageDataLoader(FragmentDataLoader):
    """Data loader that loads directly from image files (TIFF, JPG, JPEG)."""

    def __init__(self, fragment_path: str, verbose: bool = False, contrasted: bool = False):
        self.contrasted = contrasted
        # Initialize base class first to set up self.layers_dir
        super().__init__(fragment_path, verbose)

        # Get all image files and determine layer indices
        image_files = [f for f in os.listdir(self.layers_dir) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg'))]

        # Try to extract layer indices from filenames
        try:
            layer_indices = [int(os.path.splitext(f)[0]) for f in image_files]
            self.min_layer_idx = min(layer_indices)
            self.max_layer_idx = max(layer_indices)
            if self.verbose:
                logger.info(f"Layer indices range from {self.min_layer_idx} to {self.max_layer_idx}")
        except ValueError:
            # If we can't extract indices, use relative indices
            self.min_layer_idx = 0
            self.max_layer_idx = len(image_files) - 1
            if self.verbose:
                logger.info(
                    f"Could not extract layer indices from filenames. Using relative indices 0 to {self.max_layer_idx}")

        # Create LUT for contrast enhancement if needed
        self.lut = None
        if self.contrasted:
            self._init_contrast_lut()

    def _init_contrast_lut(self):
        # Check first image to determine bit depth
        img_path = self.get_layer_path(self.min_layer_idx)
        if img_path.lower().endswith(('.tif', '.tiff')):
            first_image = tifffile.imread(img_path)
        else:  # JPG/JPEG
            first_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if first_image.dtype == np.uint16:
            bit_depth = 16
        elif first_image.dtype == np.uint8:
            bit_depth = 8
        else:
            raise ValueError(f"Unsupported image dtype: {first_image.dtype}")

        control_points = [(0, 0), (128, 64), (255, 255)]
        self.lut = create_lut(control_points, bit_depth=bit_depth)
        if self.verbose:
            logger.info(f"Created contrast LUT for {bit_depth}-bit images")

    def get_fragment_dimensions(self) -> Tuple[int, int]:
        first_file = sorted(os.listdir(self.layers_dir))[0]
        img_path = os.path.join(self.layers_dir, first_file)
        if first_file.lower().endswith(('.tif', '.tiff')):
            with tifffile.TiffFile(img_path) as tif:
                height, width = tif.pages[0].shape
        else:  # JPG/JPEG
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
        return height, width

    def load_chunk(self, layer_start: int, num_layers: int,
                   start_x: int, end_x: int,
                   start_y: int = None, end_y: int = None) -> np.ndarray:
        if self.verbose:
            if num_layers == 1:
                logger.info(f"Loading layer {layer_start} (rows {start_y}:{end_y}, cols {start_x}:{end_x})")
            else:
                logger.info(
                    f"Loading layers {layer_start}-{layer_start + num_layers - 1} (rows {start_y}:{end_y}, cols {start_x}:{end_x})")
        chunk_data = []
        for i in range(layer_start, layer_start + num_layers):
            img_path = self.get_layer_path(i)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Missing layer file: {img_path}")
            if img_path.lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(img_path)[start_y:end_y, start_x:end_x]  # TIF
            else:
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[start_y:end_y, start_x:end_x]  # JPG/JPEG

            # Apply contrast enhancement if enabled
            if self.contrasted:
                # Initialize LUT if not already done
                if self.lut is None:
                    self._init_contrast_lut()
                image = apply_lut(image, self.lut)

            assert 1 < np.asarray(image).max() <= 255, "Invalid image index {}".format(i)
            chunk_data.append(image)
        result = np.stack(chunk_data, axis=0)
        if self.verbose:
            logger.info(f"Chunk shape: {result.shape}, dtype: {result.dtype}")
            logger.info(f"Chunk stats: min={result.min()}, max={result.max()}, mean={result.mean():.2f}")
        return result


class ZarrDataLoader(FragmentDataLoader):
    """Data loader that loads from a zarr store."""

    def __init__(self, fragment_path: str, zarr_store_path: Optional[str] = None, chunk_width: int = 4096,
                 verbose: bool = False, contrasted: bool = False):
        self.contrasted = contrasted
        self.verbose = verbose
        self.fragment_path = fragment_path
        self.zarr_store_path = zarr_store_path or fragment_path
        store_name = "layers-contrasted.zarr" if contrasted else "layers.zarr"
        self.zarr_path = os.path.join(self.zarr_store_path, store_name)
        self.chunk_width = chunk_width

        # Create or update store if needed.
        self._create_or_update_store(verbose=verbose)

        # Open the store once and cache it.
        self._zarr_store = zarr.open(self.zarr_path, mode='r')

        super().__init__(fragment_path, verbose)

        if 'min_layer_idx' in self._zarr_store.attrs and 'max_layer_idx' in self._zarr_store.attrs:
            self.min_layer_idx = int(self._zarr_store.attrs['min_layer_idx'])
            self.max_layer_idx = int(self._zarr_store.attrs['max_layer_idx'])
            if self.verbose:
                logger.info(f"Zarr store contains layers {self.min_layer_idx} to {self.max_layer_idx}")
        else:
            self.min_layer_idx = 0
            self.max_layer_idx = self._zarr_store.shape[0] - 1
            if self.verbose:
                logger.info(f"No layer index metadata found. Assuming layers 0 to {self.max_layer_idx}.")

    def get_fragment_dimensions(self) -> Tuple[int, int]:
        return self._zarr_store.shape[1], self._zarr_store.shape[2]

    def _create_or_update_store(self, verbose, dask=True) -> None:
        if not os.path.exists(self.zarr_path):
            logger.info(f"Zarr store not found at {self.zarr_path}. Creating it...")
            if self.contrasted:
                # If contrasted store doesn't exist, check if canonical store exists
                canonical_store_path = os.path.join(self.zarr_store_path, "layers.zarr")
                if os.path.exists(canonical_store_path):
                    if verbose:
                        logger.info(
                            f"Canonical zarr store found at {canonical_store_path}. Using it to create contrasted store...")
                        if dask:
                            self._create_contrasted_store_from_canonical_with_dask(canonical_store_path)
                        else:
                            self._create_contrasted_store_from_canonical_zarr(canonical_store_path)
                else:
                    if verbose:
                        logger.info("No canonical store found. Creating contrasted store from image files...")
                    self._create_contrasted_store_from_images()
            else:
                if verbose:
                    logger.info(
                        f"Canonical zarr store not found at {self.zarr_path}. Creating store from image files...")
                self._create_store_from_images()

    def _create_store_from_images(self, chunk_width: int = None) -> None:
        # Create normal zarr store from image files.
        layer_path = os.path.join(self.fragment_path, "layers")
        if not os.path.isdir(layer_path):
            raise ValueError(f"The directory {layer_path} does not exist.")

        image_files = [f for f in os.listdir(layer_path) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError("No image files (TIFF, JPG, JPEG) found in the 'layers' directory.")

        try:
            image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
        except ValueError:
            image_files = sorted(image_files)

        if self.verbose:
            logger.info(f"Found {len(image_files)} image files.")

        first_image_path = os.path.join(layer_path, image_files[0])
        if first_image_path.lower().endswith(('.tif', '.tiff')):
            first_image = tifffile.imread(first_image_path)
        else:  # JPG/JPEG
            first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)

        height, width = first_image.shape
        num_layers = len(image_files)

        # Determine chunk dimensions.
        cw = min(chunk_width or self.chunk_width, width)
        chunk_height = min(4096, height)

        compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        z = zarr.open(self.zarr_path, mode='w', shape=(num_layers, height, width),
                      chunks=(1, chunk_height, cw), dtype=np.uint8, compressor=compressor)
        z[0, :, :] = first_image

        for i, fname in enumerate(tqdm(image_files[1:], desc="Processing image files", unit="file"), start=1):
            img_path = os.path.join(layer_path, fname)
            if fname.lower().endswith(('.tif', '.tiff')):
                img = tifffile.imread(img_path)
            else:  # JPG/JPEG
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            z[i, :, :] = img

        z.attrs['min_layer_idx'] = 0
        z.attrs['max_layer_idx'] = num_layers - 1

        if self.verbose:
            logger.info("Normal zarr store created successfully.")

    def _create_contrasted_store_from_images(self, chunk_width: int = None, use_parallel=True) -> None:
        """
            Create a contrasted zarr store from image files.
            Applies a fixed contrast LUT (with control points [(0,0), (128,64), (255,255)]).
            Can use parallel or sequential processing.

            Parameters:
                chunk_width: Optional width of chunks for zarr storage
                use_parallel: Whether to use parallel processing (default True)
        """
        control_points = [(0, 0), (128, 64), (255, 255)]
        layer_path = os.path.join(self.fragment_path, "layers")
        if not os.path.isdir(layer_path):
            raise ValueError(f"The directory {layer_path} does not exist.")

        image_files = [f for f in os.listdir(layer_path) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError("No image files (TIFF, JPG, JPEG) found in the 'layers' directory.")

        # Extract layer indices from filenames
        try:
            layer_indices = [int(os.path.splitext(f)[0]) for f in image_files]
            min_layer_idx = min(layer_indices)
            max_layer_idx = max(layer_indices)

            # Sort files by their actual layer number
            sorted_data = sorted(zip(layer_indices, image_files))
            layer_indices = [idx for idx, _ in sorted_data]
            image_files = [f for _, f in sorted_data]

            if self.verbose:
                logger.info(
                    f"Found {len(image_files)} image files with layer indices from {min_layer_idx} to {max_layer_idx}")
        except ValueError:
            # If we can't extract indices, fall back to simple sorting and use relative indices
            image_files = sorted(image_files)
            min_layer_idx = 0
            max_layer_idx = len(image_files) - 1
            layer_indices = list(range(min_layer_idx, max_layer_idx + 1))

            if self.verbose:
                logger.info(
                    f"Could not extract layer indices from filenames. Using relative indices 0 to {max_layer_idx}.")

        # Process first image
        first_image_path = os.path.join(layer_path, image_files[0])
        if first_image_path.lower().endswith(('.tif', '.tiff')):
            first_image = tifffile.imread(first_image_path)
        else:  # JPG/JPEG
            first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)

        if first_image.dtype == np.uint16:
            bit_depth = 16
        elif first_image.dtype == np.uint8:
            bit_depth = 8
        else:
            raise ValueError("Unsupported image dtype: {}".format(first_image.dtype))

        lut = create_lut(control_points, bit_depth=bit_depth)
        contrasted_first = apply_lut(first_image, lut)
        height, width = contrasted_first.shape
        num_layers = len(image_files)
        cw = min(chunk_width or self.chunk_width, width)
        chunk_height = min(4096, height)

        logger.info(f"Creating contrasted zarr store from images with dtype {contrasted_first.dtype}")

        compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        z = zarr.open(self.zarr_path, mode='w', shape=(num_layers, height, width),
                      chunks=(1, chunk_height, cw), dtype=contrasted_first.dtype, compressor=compressor)
        z[0, :, :] = contrasted_first

        if self.verbose:
            logger.info(f"First image processed and stored. Processing remaining {len(image_files) - 1} images...")

        # Process remaining images
        if use_parallel:
            try:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                # Prepare image paths for processing
                image_paths = [os.path.join(layer_path, fname) for fname in image_files[1:]]

                with ProcessPoolExecutor() as executor:
                    # Submit tasks for layers 1 to num_layers-1 using the static method
                    futures = []
                    for img_path in image_paths:
                        futures.append(executor.submit(self._process_image_with_lut, img_path, lut))

                    # Use tqdm to show progress
                    with tqdm(total=len(futures), desc="Processing images (parallel)", unit="file") as pbar:
                        for i, future in enumerate(as_completed(futures), start=1):
                            try:
                                contrasted_img = future.result()
                                z[i, :, :] = contrasted_img
                            except Exception as e:
                                logger.error(f"Error processing image {i}: {str(e)}")
                                raise
                            pbar.update(1)

                if self.verbose:
                    logger.info("Parallel processing completed successfully.")

            except Exception as e:
                logger.error(f"Parallel processing failed: {str(e)}")
                logger.info("Falling back to sequential processing...")
                use_parallel = False

        # Sequential processing (either as primary method or fallback)
        if not use_parallel:
            for i, fname in enumerate(tqdm(image_files[1:], desc="Processing images (sequential)", unit="file"),
                                      start=1):
                try:
                    img_path = os.path.join(layer_path, fname)
                    if fname.lower().endswith(('.tif', '.tiff')):
                        img = tifffile.imread(img_path)
                    else:  # JPG/JPEG
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    contrasted_img = apply_lut(img, lut)
                    z[i, :, :] = contrasted_img
                except Exception as e:
                    logger.error(f"Error processing image {i} ({fname}): {str(e)}")
                    raise

        # Set attributes with the actual min and max layer indices
        z.attrs['min_layer_idx'] = min_layer_idx
        z.attrs['max_layer_idx'] = max_layer_idx

        if self.verbose:
            logger.info(
                f"Contrasted zarr store created successfully with layer indices {min_layer_idx}-{max_layer_idx}")

    def _create_contrasted_store_from_canonical_zarr(self, canonical_store_path: str, parallel=False) -> None:
        """
          Create a contrasted zarr store from an existing normal zarr store.
          Applies the fixed contrast LUT in parallel over all layers.
        """
        # Open the canonical store and determine bit depth.
        canonical_store = zarr.open(canonical_store_path, mode='r')
        num_layers, height, width = canonical_store.shape
        if canonical_store.dtype == np.uint16:
            bit_depth = 16
        elif canonical_store.dtype == np.uint8:
            bit_depth = 8
        else:
            raise ValueError("Unsupported dtype in normal store: {}".format(canonical_store.dtype))
        control_points = [(0, 0), (128, 64), (255, 255)]
        lut = create_lut(control_points, bit_depth=bit_depth)

        chunk_height = canonical_store.chunks[1] if canonical_store.chunks else min(4096, height)
        cw = canonical_store.chunks[2] if canonical_store.chunks else self.chunk_width

        compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
        logging.info(f"Creating zarr store with dtype {canonical_store.dtype}")
        z = zarr.open(self.zarr_path, mode='w', shape=(num_layers, height, width),
                      chunks=(1, chunk_height, cw), dtype=canonical_store.dtype, compressor=compressor)

        if parallel:
            # Helper function for parallel processing
            def process_layer(i):
                img = canonical_store[i, :, :]
                return apply_lut(img, lut)

            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(process_layer, i): i for i in range(num_layers)}
                for future in as_completed(futures):
                    i = futures[future]
                    contrasted_img = future.result()
                    z[i, :, :] = contrasted_img
        else:
            for i in tqdm(range(num_layers), desc="Processing layers", unit="layer"):
                img = canonical_store[i, :, :]
                contrasted_img = apply_lut(img, lut)
                z[i, :, :] = contrasted_img

        z.attrs['min_layer_idx'] = canonical_store.attrs.get('min_layer_idx', 0)
        z.attrs['max_layer_idx'] = canonical_store.attrs.get('max_layer_idx', num_layers - 1)
        if self.verbose:
            logger.info("Contrasted zarr store created successfully from canonical store.")

    def _create_contrasted_store_from_canonical_with_dask(self, canonical_store_path: str) -> None:
        """
        Create a contrasted zarr store from an existing canonical zarr store using Dask.
        Applies the fixed contrast LUT in parallel over all layers.
        """
        # Open the canonical store and determine bit depth.
        canonical_store = zarr.open(canonical_store_path, mode='r')
        num_layers, height, width = canonical_store.shape
        if canonical_store.dtype == np.uint16:
            bit_depth = 16
        elif canonical_store.dtype == np.uint8:
            bit_depth = 8
        else:
            raise ValueError("Unsupported dtype in canonical store: {}".format(canonical_store.dtype))

        control_points = [(0, 0), (128, 64), (255, 255)]
        lut = create_lut(control_points, bit_depth=bit_depth)

        # Create a Dask array from the canonical store.
        darr = da.from_zarr(canonical_store_path)

        # Define a function to apply the LUT to a block.
        def apply_lut_block(block):
            return np.take(lut, block)

        # Apply the LUT over the entire array using map_blocks.
        contrasted_darr = darr.map_blocks(apply_lut_block, dtype=canonical_store.dtype)

        # Start a timer and use Dask's ProgressBar to show progress.
        start_time = time.time()
        logger.info("Starting contrasted store creation using Dask...")
        with ProgressBar():
            contrasted_darr.to_zarr(self.zarr_path, overwrite=True)
        elapsed_time = time.time() - start_time
        logger.info(f"Contrasted zarr store creation completed in {elapsed_time:.2f} seconds.")

        canonical_attrs = {
            'min_layer_idx': canonical_store.attrs.get('min_layer_idx', 0),
            'max_layer_idx': canonical_store.attrs.get('max_layer_idx', num_layers - 1)
        }
        new_store = zarr.open(self.zarr_path, mode='a')
        new_store.attrs.update(canonical_attrs)

        if self.verbose:
            logger.info("Contrasted zarr store created successfully from canonical store using Dask.")

    def load_chunk(self, layer_start: int, num_layers: int,
                   start_x: int, end_x: int,
                   start_y: int = None, end_y: int = None) -> np.ndarray:
        if self.verbose:
            if num_layers == 1:
                logger.info(f"Loading layer {layer_start} from Zarr (rows {start_y}:{end_y}, cols {start_x}:{end_x})")
            else:
                logger.info(
                    f"Loading layers {layer_start}-{layer_start + num_layers - 1} from Zarr (rows {start_y}:{end_y}, cols {start_x}:{end_x})")
        zarr_start_idx = layer_start - self.min_layer_idx
        zarr_end_idx = zarr_start_idx + num_layers

        if zarr_start_idx < 0:
            raise ValueError(f"Layer index {layer_start} is below the minimum available layer ({self.min_layer_idx})")
        if layer_start + num_layers - 1 > self.max_layer_idx:
            raise ValueError(
                f"Requested layers up to {layer_start + num_layers - 1}, but only layers up to {self.max_layer_idx} are available")

        result = self._zarr_store[zarr_start_idx:zarr_end_idx, start_y:end_y, start_x:end_x]
        if result.dtype != np.uint8:
            result = result.astype(np.uint8)

        if self.verbose:
            logger.info(f"Zarr chunk shape: {result.shape}, dtype: {result.dtype}")
            logger.info(f"Zarr chunk stats: min={result.min()}, max={result.max()}, mean={result.mean():.2f}")
        return result


def create_data_loader(
        fragment_path: str,
        use_zarr: bool = True,
        zarr_store_path: Optional[str] = None,
        verbose: bool = False,
        chunk_width: int = 4096,
        contrasted: bool = False
) -> FragmentDataLoader:
    zarr_store_path = zarr_store_path or fragment_path
    if use_zarr:
        return ZarrDataLoader(
            fragment_path=fragment_path,
            zarr_store_path=zarr_store_path,
            chunk_width=chunk_width,
            verbose=verbose,
            contrasted=contrasted
        )
    return ImageDataLoader(fragment_path=fragment_path, verbose=verbose, contrasted=contrasted)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    frag_path = os.path.join(os.getcwd(), "data", f"scroll5", "fragments", "02110815")
    data_loader = create_data_loader(fragment_path=frag_path, use_zarr=True, contrasted=True)
    data_loader.plot_sample(
        layer_start=8, x_range=(410000, 420112),
        output_path="data/scroll5/fragments/02110815/contrast_sample.tif"
    )

    frag_path = os.path.join(os.getcwd(), "data", f"scroll5", "fragments", "20241108120732")
    data_loader = create_data_loader(fragment_path=frag_path, use_zarr=True, contrasted=True)
    data_loader.plot_sample(square_size=1000)
