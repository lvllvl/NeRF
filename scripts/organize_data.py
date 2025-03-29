import os
import shutil

# Define the paths
base_dir = 'data/'  # Your current data folder
llff_dir = os.path.join(base_dir, 'llff')
synthetic_dir = os.path.join(base_dir, 'synthetic')

# Ensure target directories exist
os.makedirs(llff_dir, exist_ok=True)
os.makedirs(synthetic_dir, exist_ok=True)

# Step 1: Move LLFF data to the appropriate directory
fern_dir = os.path.join(base_dir, 'nerf_llff_data', 'fern')
if os.path.exists(fern_dir):
    shutil.move(fern_dir, llff_dir)
    print(f"Moved LLFF 'fern' data to {llff_dir}")
    
# Step 2: Move synthetic data (e.g., 'lego') to the appropriate directory
lego_dir = os.path.join(base_dir, 'nerf_synthetic', 'lego')
if os.path.exists(lego_dir):
    shutil.move(lego_dir, synthetic_dir)
    print(f"Moved synthetic 'lego' data to {synthetic_dir}")

# Clean up empty directories
if os.path.exists(os.path.join(base_dir, 'nerf_llff_data')):
    shutil.rmtree(os.path.join(base_dir, 'nerf_llff_data'))
if os.path.exists(os.path.join(base_dir, 'nerf_synthetic')):
    shutil.rmtree(os.path.join(base_dir, 'nerf_synthetic'))

print("Data organization complete!")
