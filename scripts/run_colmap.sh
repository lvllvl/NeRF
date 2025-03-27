#!/bin/bash

# Set Qt to offscreen mode for headless environments
export QT_QPA_PLATFORM=offscreen

# (Optional) Set XDG_RUNTIME_DIR to suppress related warnings
export XDG_RUNTIME_DIR=/tmp/runtime-vscode

# List the image directory to verify the correct path
echo "Listing image directory:"
ls /workspaces/NeRF/colmap_images

# Run COLMAP feature extractor in CPU mode (disable GPU usage)
colmap feature_extractor \
  --database_path /workspaces/NeRF/colmap.db \
  --image_path /workspaces/NeRF/colmap_images \
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
  --image_path /workspaces/NeRF/colmap_images \
  --output_path /workspaces/NeRF/sparse
