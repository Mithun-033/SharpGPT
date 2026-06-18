#!/bin/bash

cd SharpGPT

echo "Downloading pre-trained data..."

python train/data.py

echo "Pre-trained data downloaded successfully."

#------------------------------------------------------------------------------------#
echo "training tokenizer..."

chmod +x tokenizer.sh
./tokenizer.sh

echo "Starting training process..."
python train/train.py





