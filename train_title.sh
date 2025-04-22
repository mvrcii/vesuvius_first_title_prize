#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate scroll5-title

BASE_PATH="data"
FRAGMENT_ID="03192025"
CHUNK_IDS="1,3,4,9,11,13,15,19,20,21,24,25,26,27,28,29"
CONFIG_PATH="configs/ft_no_title.py"


# Download data
python scripts/download_fragments.py --fragment "$FRAGMENT_ID"

# Chunking and pre-processing
python scripts/fragment_splitter.py "$FRAGMENT_ID" --scroll-id "5" --base-path "$BASE_PATH" -ch "$CHUNK_IDS" --contrasted

# Creating the dataset
python scripts/create_dataset.py "$CONFIG_PATH" --ide_is_closed

# Training (defaults device to cuda)
python scripts/train.py "$CONFIG_PATH"