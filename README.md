# Scroll 5 Title Submission - Team Wuesuv

<img src="https://github.com/user-attachments/assets/4c4596c5-343c-46bf-ae23-a1431b441cd3" width="500">

### Quickstart Inference
Prerequisites: 
- **Hardware**: min. 24 GB VRAM (We used an RTX 4090), 96GB RAM, 50GB disk space
- **Software**: Python >= 3.8 and working conda install

1. Clone the repository
2. Download the [checkpoint](https://drive.google.com/file/d/1OTMnO7bgPQRUlzQZ2m7dd924FEwFDdQz/view?usp=drive_link) and place it in ``checkpoints/scroll5/warm-planet-193-unetr-sf-b3-250417-171532``
3. Execute the following command from the root directory. This will set up the conda environment with the correct python version and install the required dependencies.
```
python init_env.py
```
3. Activate the conda environment with
```
conda activate scroll5-title
```

5. Run the following script to download the required layers, preprocess them and finally run inference on the title chunk. Note that the resulting image is flipped horizontally (and must be flipped to match our submission).
```
./infer_title.sh
```
6. The results directory where the predictions will be saved will be printed to the console. It will contain 2 subdirectories. `visualizations` and `npy_files`.
To reproduce the exact image we subbmitted as our main submission, run our `scripts/overlay_viewer.py` UI, and select the results npy_files directory. Then select `horizontal flip`, `average` and set `boost` to `2.9`.

# Supplementary Info
### Training data
We used the following two VC3D auto-segmentations as training data: [02110815](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s5/autogens/02110815/) and 
[03192025](https://dl.ash2txt.org/community-uploads/bruniss/scrolls/s5/autogens/03192025/).

Our `fragment_splitter.py` script splits their large layer files into chunks with a width of 10,000 pixels and applies contrasting. We applied contrasting to all the fragment files, as we found it made it easier to visually detect very subtle ink crackles during labeling, and seemed to boost performance of our segmentation model.  

Therefore, for inference, our model also requires the **contrasted** layer files.


![contrast_compare](https://github.com/user-attachments/assets/d7e01562-6210-48e7-9e86-fa08e8da4b52)

Labels were iteratively refined and cleaned over various rounds of training and inference, starting with 02110815 and continuing with 03192025. Our final labels per chunk can be found in their respective subdirectories within `data`, and each consists of two files: an ink label (`label.png`) and an ignore mask (`ignore.png`).

We use the ignore mask to mark areas where the model's predictions were uncertain in the previous run. Rather than labeling an uncertain area as ink or no-ink, we simply cover it with the ignore mask, completely removing any covered pixels from the loss calculation — thereby avoiding the propagation of incorrect labels.

The image below shows an example of this ignore mask. Instead of completing partial letters by hand (with ink labels that were not present in the prediction), we cover incomplete letters with the ignore mask (red) — effectively adding no label to those pixels and allowing the model to figure it out on its own.

![ignore_smol](https://github.com/user-attachments/assets/c336ea44-81b3-4497-853c-93353105282d)

### Model General
Our model was trained on auto-segmentations that consist of **21** layers in total, taking in the top 16 (index 5 through 20) as input, and hence likely does not translate to the traditional 65 layer segmentations, however we didn't test this ablation. Note that all input data for inference **has** to be contrasted with the logic implemented in our `fragment_splitter.py`.

