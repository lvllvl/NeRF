#!/bin/bash

#!/bin/bash

export PATH="$(pwd)/colmap/build/src/colmap/exe:$PATH"
echo "Using COLMAP from: $(which colmap)"
colmap -h | head -n 5

# Check for COLMAP
if ! command -v colmap &> /dev/null; then
  echo "❌ Error: COLMAP not found."
  exit 1
fi
export QT_QPA_PLATFORM=offscreen


# Check for COLMAP
if ! command -v colmap &> /dev/null; then
  echo "❌ Error: COLMAP not found."
  exit 1
fi

# Set paths
IMAGES_DIR="data/raw"
OUTPUT_DIR="data/colmap_output"
DB_PATH="$OUTPUT_DIR/database.db"
SPARSE_DIR="$OUTPUT_DIR/sparse"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SPARSE_DIR"

# Step 1: Feature extraction
colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 0

# Step 2: Feature matching
colmap exhaustive_matcher \
    --database_path "$DB_PATH" \
    --SiftMatching.use_gpu 0

# Step 3: Sparse reconstruction
colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE_DIR"

# Step 4: Optional model converter
colmap model_converter \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$SPARSE_DIR/0" \
    --output_type TXT

echo "😝😝😝😝😝😝😝😝😝😝😝😝\n COLMAP reconstruction complete."
