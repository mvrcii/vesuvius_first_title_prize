import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, Frame, Checkbutton, BooleanVar, DoubleVar, StringVar, Radiobutton
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.transform import resize
import os


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Directory Image Processor")
        self.root.geometry("800x700")

        self.numpy_files = []
        self.loaded_arrays = []
        self.composite_data = None
        self.downsampled_data = None
        self.processed_data = None

        # Create UI components
        self.create_ui()

    def create_ui(self):
        # Top frame for controls
        control_frame = Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Load button
        load_button = Button(control_frame, text="Select Directory", command=self.select_directory)
        load_button.pack(side=tk.LEFT, padx=5)

        # Save button
        save_button = Button(control_frame, text="Save Processed Image", command=self.save_processed_image)
        save_button.pack(side=tk.LEFT, padx=5)

        # Composite method selection
        composite_frame = Frame(self.root)
        composite_frame.pack(fill=tk.X, padx=10, pady=5)

        composite_label = Label(composite_frame, text="Composite Method:")
        composite_label.pack(side=tk.LEFT, padx=5)

        self.composite_var = StringVar(value="average")
        average_radio = Radiobutton(composite_frame, text="Average", variable=self.composite_var,
                                    value="average", command=self.create_composite)
        average_radio.pack(side=tk.LEFT, padx=5)

        max_radio = Radiobutton(composite_frame, text="Maximum", variable=self.composite_var,
                                value="maximum", command=self.create_composite)
        max_radio.pack(side=tk.LEFT, padx=5)

        # Effects frame
        effects_frame = Frame(self.root)
        effects_frame.pack(fill=tk.X, padx=10, pady=5)

        # Invert colors checkbox
        self.invert_var = BooleanVar(value=False)
        invert_check = Checkbutton(effects_frame, text="Invert Colors", variable=self.invert_var,
                                   command=self.process_image)
        invert_check.pack(side=tk.LEFT, padx=20)

        # Horizontal flip checkbox
        self.flip_h_var = BooleanVar(value=False)
        flip_h_check = Checkbutton(effects_frame, text="Horizontal Flip", variable=self.flip_h_var,
                                   command=self.process_image)
        flip_h_check.pack(side=tk.LEFT, padx=20)

        # Boost value controls
        boost_frame = Frame(effects_frame)
        boost_frame.pack(side=tk.LEFT, padx=20)

        boost_label = Label(boost_frame, text="Boost Factor:")
        boost_label.pack(side=tk.LEFT)

        self.boost_var = DoubleVar(value=1.0)
        boost_scale = Scale(boost_frame, variable=self.boost_var, from_=1.0, to=10.0,
                            resolution=0.1, orient=tk.HORIZONTAL, length=200,
                            command=lambda x: self.process_image())
        boost_scale.pack(side=tk.LEFT)

        # Frame for image display
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status label
        self.status_label = Label(self.root, text="Ready. Please select a directory with numpy arrays.")
        self.status_label.pack(pady=10)

    def select_directory(self):
        """Select a directory containing numpy arrays"""
        dir_path = filedialog.askdirectory()

        if not dir_path:
            return

        try:
            # Find all numpy files in the directory
            self.numpy_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.npy')]

            if not self.numpy_files:
                self.status_label.config(text=f"No numpy files found in {dir_path}")
                return

            # Load all numpy arrays
            self.loaded_arrays = []
            shapes = []

            for file_path in self.numpy_files:
                try:
                    array = np.load(file_path)
                    self.loaded_arrays.append(array)
                    shapes.append(array.shape)
                except Exception as e:
                    self.status_label.config(text=f"Error loading {os.path.basename(file_path)}: {str(e)}")

            if not self.loaded_arrays:
                self.status_label.config(text="No valid numpy arrays found.")
                return

            # Check if all arrays have the same shape
            if len(set(str(shape) for shape in shapes)) > 1:
                self.status_label.config(
                    text="Warning: Not all arrays have the same shape. Using the first array's shape.")

            # Create composite image
            self.create_composite()

        except Exception as e:
            self.status_label.config(text=f"Error loading directory: {str(e)}")

    def create_composite(self):
        """Create a composite image using either average or maximum method"""
        if not self.loaded_arrays:
            return

        try:
            # Ensure all arrays have the same shape (resize if needed)
            reference_shape = self.loaded_arrays[0].shape
            normalized_arrays = []

            for array in self.loaded_arrays:
                if array.shape != reference_shape:
                    # Resize to match the first array's shape
                    resized = resize(array, reference_shape, anti_aliasing=True)
                    normalized_arrays.append(resized)
                else:
                    # Normalize to 0-1 range if needed
                    if array.max() > 1.0:
                        normalized = array / array.max()
                    else:
                        normalized = array
                    normalized_arrays.append(normalized)

            # Create composite based on selected method
            method = self.composite_var.get()

            if method == "average":
                self.composite_data = np.mean(normalized_arrays, axis=0)
                method_name = "Average"
            else:  # maximum
                self.composite_data = np.max(normalized_arrays, axis=0)
                method_name = "Maximum"

            self.status_label.config(
                text=f"Created {method_name} composite from {len(self.loaded_arrays)} arrays with shape {reference_shape}")

            # Downsample the composite image
            self.downsample()

            # Process and display the image
            self.process_image()

        except Exception as e:
            self.status_label.config(text=f"Error creating composite: {str(e)}")

    def save_processed_image(self):
        """Save the processed image as a PNG file"""
        if self.processed_data is None:
            self.status_label.config(text="No processed image to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")])

        if not file_path:
            return

        try:
            # Ensure image data is in the correct range for PNG (0-1)
            img_data = np.clip(self.processed_data, 0, 1)

            # Convert to 8-bit format (0-255)
            img_data = (img_data * 255).astype(np.uint8)

            # Save using matplotlib
            plt.imsave(file_path, img_data, cmap='gray')
            self.status_label.config(text=f"Processed image saved to {file_path}")
        except Exception as e:
            self.status_label.config(text=f"Error saving file: {str(e)}")

    def downsample(self):
        """Downsample the image by a factor of 4"""
        if self.composite_data is None:
            return

        # Calculate new dimensions
        new_shape = tuple(dim // 4 for dim in self.composite_data.shape)

        # Downsample using resize from skimage
        self.downsampled_data = resize(self.composite_data, new_shape, anti_aliasing=True)

        self.status_label.config(
            text=f"Original: {self.composite_data.shape}, Downsampled: {self.downsampled_data.shape}")

    def boost_values(self, image, boost_factor):
        """Boost low values while keeping max at 1.0"""
        # Make a copy to avoid modifying the original
        boosted = image.copy()

        # Only boost values less than 1
        mask = boosted < 1.0
        boosted[mask] = boosted[mask] * boost_factor

        # Cap at 1.0
        boosted[boosted > 1.0] = 1.0

        return boosted

    def process_image(self):
        """Apply selected processing to the image and display it"""
        if self.downsampled_data is None:
            return

        # Start with the downsampled data
        self.processed_data = self.downsampled_data.copy()

        # Apply horizontal flip if needed
        if self.flip_h_var.get():
            self.processed_data = np.fliplr(self.processed_data)

        # Apply boost if needed
        boost_factor = self.boost_var.get()
        if boost_factor > 1.0:
            self.processed_data = self.boost_values(self.processed_data, boost_factor)

        # Apply invert if needed
        if self.invert_var.get():
            self.processed_data = 1.0 - self.processed_data

        # Display the processed image
        self.display_image()

    def display_image(self):
        """Display only the processed image"""
        if self.processed_data is None:
            return

        # Clear previous plot
        self.ax.clear()

        # Plot processed image
        self.ax.imshow(self.processed_data, cmap='gray')

        # Create title with current settings
        composite_method = "Average" if self.composite_var.get() == "average" else "Maximum"
        title = f"{composite_method} Composite - {len(self.loaded_arrays)} Images"

        if self.invert_var.get():
            title += " (Inverted)"
        if self.flip_h_var.get():
            title += " (H-Flipped)"
        if self.boost_var.get() > 1.0:
            title += f" (Boost: {self.boost_var.get():.1f}x)"
        self.ax.set_title(title)
        self.ax.axis('off')

        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()