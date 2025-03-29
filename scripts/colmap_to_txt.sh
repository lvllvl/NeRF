#!/bin/bash

# Run this after: run_colmap.sh, or just combine these 2 scripts
# Make sure the output directory exists
mkdir -p sparse/0_txt

colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse/0_txt \
    --output_type TXT
