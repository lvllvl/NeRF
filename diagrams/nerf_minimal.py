# this is a scaffold version of NeRF 

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 📦 1. Ray Sampling
#  How 2D pixels become 3D rays 
# ------------------------------

def get_rays(camera_pose, intrinsics, H, W):
    """
    Given camera pose and intrinsics, generate rays for each pixel.
    """
    # Placeholder logic — normally uses meshgrid, intrinsics
    rays_o = torch.zeros(H, W, 3)  # origin
    rays_d = torch.ones(H, W, 3)   # direction
    return rays_o, rays_d

# How to slice rays and sample the 3D space
def sample_points_along_rays(rays_o, rays_d, near, far, N_samples):
    """
    Uniformly sample 3D points along each ray.
    """
    t_vals = torch.linspace(near, far, steps=N_samples)  # [N_samples]
    # Broadcast to match rays
    samples = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.view(1, 1, -1, 1)
    return samples  # [H, W, N_samples, 3]

# ------------------------------
# 🔁 2. Positional Encoding
# ------------------------------
# why high-frequency features help MLPs learn the fine details 
def positional_encoding(x, num_freqs=10):
    """
    Apply sine/cosine positional encoding to input.
    """
    encodings = [x]
    for i in range(num_freqs):
        for fn in [torch.sin, torch.cos]:
            encodings.append(fn((2.0 ** i) * x))
    return torch.cat(encodings, dim=-1)  # [..., num_freqs*2*dim]

# ------------------------------
# 🧠 3. NeRF Model (MLP)
# ------------------------------

class NeRF(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 1 for density (σ), 3 for RGB
        )
    # How MLP turns 3D points into RGP and density 
    def forward(self, x):
        return self.layers(x)

# ------------------------------
# 🧮 4. Volume Rendering
# ------------------------------
# how NeRF mixes predictions into 1 pixel
def volume_rendering(raw, z_vals):
    """
    Combine raw NeRF outputs (density + RGB) into final pixel color.
    """
    sigma = F.relu(raw[..., 0])  # [H, W, N_samples]
    rgb = torch.sigmoid(raw[..., 1:])  # [H, W, N_samples, 3]

    deltas = z_vals[..., 1:] - z_vals[..., :-1]
    deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[..., :1])], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * deltas)  # [H, W, N_samples]
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = T * alpha  # [H, W, N_samples]

    color = (weights.unsqueeze(-1) * rgb).sum(dim=-2)  # [H, W, 3]
    return color

# ------------------------------
# 🧪 5. Loss Function
# ------------------------------

def compute_loss(pred, target):
    return F.mse_loss(pred, target)

# ------------------------------
# 🚀 Main Training Loop
# ------------------------------
# how all pieces link in backprop
def train_step(nerf_model, optimizer, rays_o, rays_d, target_rgb):
    """
    One step of training.
    """
    samples = sample_points_along_rays(rays_o, rays_d, near=2.0, far=6.0, N_samples=64)
    pe = positional_encoding(samples)  # [H, W, N_samples, D]
    B, H, N, D = pe.shape

    flat_pe = pe.view(-1, D)
    raw = nerf_model(flat_pe).view(B, H, N, 4)  # [H, W, N_samples, 4]

    z_vals = torch.linspace(2.0, 6.0, steps=64).expand(H, W, 64)
    pred_rgb = volume_rendering(raw, z_vals)

    loss = compute_loss(pred_rgb, target_rgb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# ------------------------------
# 🧪 Example Driver (pseudo)
# ------------------------------

def main():
    H, W = 100, 100
    rays_o, rays_d = get_rays(None, None, H, W)
    target_rgb = torch.rand(H, W, 3)  # placeholder ground truth

    pe_dim = 3 * (1 + 2 * 10)  # 3 coords × 2 functions × 10 frequencies
    nerf_model = NeRF(pe_dim)
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=5e-4)

    for step in range(1000):
        loss = train_step(nerf_model, optimizer, rays_o, rays_d, target_rgb)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
