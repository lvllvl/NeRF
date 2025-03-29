#!/bin/bash

# Set Qt to offscreen mode for headless environments
export QT_QPA_PLATFORM=offscreen

# Set XDG_RUNTIME_DIR to suppress related warnings
export XDG_RUNTIME_DIR=/tmp/runtime-vscode

# Remove old database stuff and sparse output, if it exists
rm colmap.db
rm -rf sparse/

# List the image directory to verify the correct path
echo "Listing image directory:"
ls /workspaces/NeRF/data/raw/plant/images/

# Run COLMAP feature extractor in CPU mode (disable GPU usage)
colmap feature_extractor \
  --database_path /workspaces/NeRF/colmap.db \
  --image_path /workspaces/NeRF/data/raw/plant/images/ \
  --SiftExtraction.use_gpu 0

# Run COLMAP exhaustive matcher in CPU mode
colmap exhaustive_matcher \
  --database_path /workspaces/NeRF/colmap.db \
  --SiftMatching.use_gpu 0

# Ensure the output directory exists
mkdir -p /workspaces/NeRF/sparse

# Run COLMAP mapper to generate the sparse reconstruction
colmap mapper \
  --database_path /workspaces/NeRF/colmap.db \
  --image_path /workspaces/NeRF/data/raw/plant/images/ \
  --output_path /workspaces/NeRF/sparse

# Export the txt file(s)
# Make sure the output directory exists
mkdir -p sparse/0_txt

colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse/0_txt \
    --output_type TXT