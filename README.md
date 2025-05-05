# Vesuvius First Title Prize
<img src="https://github.com/user-attachments/assets/eba007fc-767f-4fce-b929-aa6ee0468039" width="500">

*Winning First Title Prize Submission by Micha Nowak and Marcel Roth* 


## Prerequisites
- **Hardware**: min. 24 GB VRAM (We used an RTX 4090), 96GB RAM, 50GB disk space
- **Software**: Python >= 3.8 and working conda install

## Quickstart - Inference
1. Clone the repository
2. Download the [checkpoint](https://drive.google.com/file/d/1OTMnO7bgPQRUlzQZ2m7dd924FEwFDdQz/view?usp=drive_link) and place it in ``checkpoints/scroll5/warm-planet-193-unetr-sf-b3-250417-171532``
3. Execute the following command from the root directory. This will set up the conda environment with the correct python version and install the required dependencies.
```
python init_env.py
```
4. Activate the conda environment with
```
conda activate scroll5-title
```
5. Run the following script to download the required layers, preprocess them and finally run inference on the title chunk. Note that the resulting image is flipped horizontally (and must be flipped to match our submission).
```
./infer_title.sh
```
6. The results directory where the predictions will be saved will be printed to the console. It will contain 2 subdirectories. `visualizations` and `npy_files`.
To reproduce the exact image we submitted, run our `scripts/overlay_viewer.py` UI, and select the resulting `npy_files` directory. Then select `horizontal flip`, `average` and set `boost` to `3` (Make sure invert colors is unchecked).

## Quickstart - Training
1. Clone the repository
2. Execute the following command from the root directory. This will set up the conda environment with the correct python version, install torch and all required packages, and finally installs our phoenix package.
```
python init_env.py
```
3. Activate the conda environment
```
conda activate scroll5-title
```
4. Run the following commands one by another. We download, chunk and pre-process the required fragment. Then we create the training dataset and start to train the model on the fragment chunks specified by the config. 
```shell
python scripts/download_fragments.py --fragment 03192025
python scripts/fragment_splitter.py 03192025 --scroll-id 5 -ch 1,3,4,9,11,13,15,19,20,21,24,25,26,27,28,29 --contrasted
python scripts/create_dataset.py configs/ft_no_title.py --ide_is_closed
python scripts/train.py configs/ft_no_title.py
```
Important notes: 
- `download_fragments.py` is as of now hardcoded for Sean's autosegmentation fragments. The script also skips existing files.
- `fragment_splitter.py` is user-friendly as it only creates the chunks as given by the command. It skips the creation of already existing chunk files and if interrupted while processing, we continue where we left off. 
- `create_dataset.py` is written very efficiently, meaning that it consumes almost 100% of all CPU cores. We recommend to execute it from within a console and not within an IDE as this might crash the IDE.
- We trained the submitted model for a total of 14 epochs.


## Supplementary Info
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

