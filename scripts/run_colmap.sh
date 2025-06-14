#!/bin/bash

# Setting paths 
IMAGES_DIR="data/raw"
OUTPUT_DIR="data/colmap_output"
DB_PATH="$OUTPUT_DIR/database.db"
SPARSE_DIR="$OUTPUT_DIR/sparse"

# Create output directory
mdir -p "$OUTPUT_DIR"
mdir -p "$SPARSE_DIR"

# Step 1: Feature extraction
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR"
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1

# Step 2: Feature matching (exhaustive is simplest; sequential is better for video)
colmap exhaustive_matcher \
    --database_path "$DB_PATH"
    --SiftMatching.use_gpu 1

# Step 3: Sparse reconstruction (Structure from Motion)
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE_DIR"

# Step 4: Optionally convert to text for inspection
colmap model_converter \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$SPARSE_DIR/0" \
    --output_type TXT

echo "😝😝😝😝😝😝😝😝😝😝😝😝\n COLMAP reconstruction complete."
