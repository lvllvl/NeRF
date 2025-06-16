#!/bin/bash

#!/bin/bash

# ========== CHECK COLMAP ==========

# If using headless COLMAP from source build
COLMAP_BIN="/content/colmap/build/colmap"

if [ ! -f "$COLMAP_BIN" ]; then
  echo "❌ COLMAP not found at $COLMAP_BIN"
  echo "👉 Please run the install cell in your Colab notebook first."
  exit 1
fi

# ========== SET PATHS ==========

IMAGES_DIR="data/raw"
OUTPUT_DIR="data/colmap_output"
DB_PATH="$OUTPUT_DIR/database.db"
SPARSE_DIR="$OUTPUT_DIR/sparse"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SPARSE_DIR"

# ========== STEP 1: FEATURE EXTRACTION ==========

"$COLMAP_BIN" feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --ImageReader.single_camera 1

# ========== STEP 2: MATCHING ==========

"$COLMAP_BIN" sequential_matcher \
    --database_path "$DB_PATH"

# ========== STEP 3: SPARSE RECONSTRUCTION ==========

"$COLMAP_BIN" mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMAGES_DIR" \
    --output_path "$SPARSE_DIR"

# ========== STEP 4: OPTIONAL TXT CONVERSION ==========

"$COLMAP_BIN" model_converter \
    --input_path "$SPARSE_DIR/0" \
    --output_path "$SPARSE_DIR/0" \
    --output_type TXT

echo "😝😝😝😝😝😝😝😝😝😝😝😝\n COLMAP reconstruction complete."
