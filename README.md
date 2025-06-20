# NeRF
Implementation of NeRF

A high level overview  of the project. 

```bash
nerf_project/
├── data/                   # raw and processed data
│   └── raw/                # your photos
│   └── colmap_output/      # output from COLMAP
├── scripts/
│   └── run_colmap.sh       # runs COLMAP CLI steps
├── utils/
│   ├── colmap_utils.py     # ✅ your functions (list, visualize, convert, etc.)
│   └── read_write_model.py # ✅ COLMAP binary reader
├── dataset.py              # dataset loader (uses transforms.json)
├── model.py                # NeRF model
├── train.py                # training loop
├── render.py               # volume rendering logic
├── main.py                 # CLI or entrypoint
└── README.md
```


# Diagrams for NeRF

```bash
nerf_minimal.py
├── main()
│   ├── get_rays()
│   │   ├── returns rays_o (origin)
│   │   └── returns rays_d (direction)
│   ├── sample_points_along_rays()
│   │   └── uses rays_o, rays_d to create 3D sample points
│   ├── positional_encoding()
│   │   └── encodes 3D points into high-freq features
│   ├── NeRF.forward()
│   │   └── predicts [σ, R, G, B] for each 3D point
│   ├── volume_rendering()
│   │   ├── computes alpha from σ
│   │   ├── blends RGB values using weights
│   │   └── outputs final RGB pixel
│   ├── compute_loss()
│   │   └── compares pred_rgb vs target_rgb
│   └── optimizer.step()
│       └── updates model weights
```

## Interpretation of the tree

### Main
Main() is the trunk. Then we branch into 
* rays
* -> samples
* -> MLP
* -> volume rendering
* -> loss 


If you want to "walk" the data through the tree, you could: 
- Camera -> get_rays -> sample_points -> encode -> MLP -> render -> pixel 

This is a linear walk-through of the model.

## Task-Level Overview (Sequential Order)

```bash
NeRF Project: "House Plant"
├── ✅ 1. Dataset Collection
│   └── Capture photos of house plant from multiple views
│       (Done ✅)
│
├── ⏳ 2. COLMAP Reconstruction
│   ├── Run COLMAP to generate:
│   │   ├── camera poses (intrinsics + extrinsics)
│   │   └── sparse/dense point cloud
│   ├── Export data in NeRF-compatible format (e.g., transforms.json)
│   └── 📁 Output: poses + images folder
│
├── 🔲 3. Project Scaffolding
│   ├── Setup repo: `nerf_project/`
│   ├── Add `nerf_minimal.py` with placeholders
│   ├── Organize file layout for modularity
│   └── Create `README.md` + config file
│
├── 🧱 4. Core Modules (scaffold)
│   ├── ray_sampling.py
│   ├── positional_encoding.py
│   ├── model.py (NeRF MLP)
│   ├── render.py (volume renderer)
│   ├── dataset.py (loads COLMAP + images)
│   └── train.py (training loop)
│
├── 🧪 5. Training & Evaluation
│   ├── Train on house plant dataset
│   ├── Save checkpoints
│   ├── Evaluate PSNR vs ground truth
│   └── Visualize: render novel views
│
└── 🎨 6. Visualization & Export
    ├── Render a video rotating around plant
    ├── Compare input views vs synthesized views
    └── Export images/video
```


### COLMAP Task Tree

```bash
COLMAP Reconstruction
├── Install COLMAP
├── Import images (JPEG or PNG)
├── Run feature extraction
├── Run matching (sequential or exhaustive)
├── Run sparse reconstruction (SfM)
├── Optionally run dense reconstruction
├── Export camera poses
│   ├── Convert to transforms.json (NeRF format)
│   └── Validate extrinsics match image order
└── Output folder
    ├── images/
    └── transforms.json (or equivalent)
```

#### Questions

-  