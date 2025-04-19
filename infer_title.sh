#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate scroll5-title

FRAGMENT_ID="03192025"
CHUNK_ID=2

FRAGMENT_CHUNK_ID="$FRAGMENT_ID/parts_contrasted/${FRAGMENT_ID}_$(printf "%02d" $CHUNK_ID)"
FRAGMENT_CHUNK_PATH="data/scroll5/fragments/${FRAGMENT_CHUNK_ID}"
FRAGMENT_PATH="data/scroll5/fragments/$FRAGMENT_ID"
CKPT_PATH="checkpoints/scroll5/warm-planet-193-unetr-sf-b3-250417-171532/best-checkpoint-epoch=13-val_iou=0.71.ckpt"
START_LAYER_IDX=3
END_LAYER_IDX=5
TTA_LEVEL=3
STRIDE=8


python scripts/download_fragments.py --fragment "$FRAGMENT_ID"

python scripts/fragment_splitter.py "$FRAGMENT_ID" -ch "$CHUNK_ID" --contrasted

# Start inference
echo "Starting inference on fragment $FRAGMENT_ID chunk $CHUNK_ID..."
python scripts/inference.py "$FRAGMENT_CHUNK_PATH" "$CKPT_PATH" --start_layer "$START_LAYER_IDX" --end_layer "$END_LAYER_IDX" --tta "$TTA_LEVEL" --stride "$STRIDE" --title_crop
