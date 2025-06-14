#!/bin/bash

# Setting paths 
IMAGES_DIR="data/raw"
OUTPUT_DIR="data/colmap_output"
DB_PATH="$OUTPUT_DIR/database.db"
SPARSE_DIR="$OUTPUT_DIR/sparse"

# Create output directory
mdir -p "$OUTPUT_DIR"
mdir -p "$SPARSE_DIR"

# Feature extraction
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR"

# Feature matching (exhaustive is simplest; sequential is better for video)
colmap exhaustive_matcher \
    --database_path "$DB_PATH"

# Sparse reconstruction (Structure from Motion)
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE_DIR"

# Optionally convert to text for inspection
colmap model_converter \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$SPARSE_DIR/0" \
    --output_type TXT

echo "😝😝😝😝😝😝😝😝😝😝😝😝\n COLMAP reconstruction complete."
