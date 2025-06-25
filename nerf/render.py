import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Dict, Tuple

def volume_rendering(
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    
    """
    Discrete approximation of the NeRF volume-rendering integral.
    Convert per-sample color and density into a single RGB, depth, opacity, and weights via NeRF integral.

    Continuous form
    ---------------
    For a ray **r**(t)=**o**+t **d**, the expected pixel colour is

        C(r) = ∫_{t_n}^{t_f} T(t) · σ(t) · c(t) dt ,
        T(t) = exp(-∫_{t_n}^{t} σ(s) ds) ,

    where c(t)∈[0,1]^3 is emitted RGB and σ(t)≥0 is volume density.

    Discrete implementation
    -----------------------
    Given *N_samples* stratified (or importance) depths **t**₀…**t**_{N-1}:

    1. **Distances** between adjacent samples  
       Δ_i = t_{i+1} - t_i (last Δ padded with a large number).
    2. **Opacity** per sample  
       α_i = 1 - exp(−σ_i · Δ_i).
    3. **Transmittance** to sample *i*  
       T_i = Π_{j<i} (1 - α_j).  Implemented via `torch.cumprod`.
    4. **Weights**  
       w_i = T_i · α_i  (probability mass at sample *i*).
    5. **Accumulate**  
       • RGB = Σ w_i · c_i                 -> (N_rays, 3)  
       • Depth = Σ w_i · t_i               -> (N_rays,)  
       • Acc = Σ w_i                       -> (N_rays,)  (1 - background transmittance)

    Parameters
    ----------
    rgb   : (N_rays, N_samples, 3) torch.Tensor
        Predicted colours in *linear* [0, 1] space from the (coarse or fine) NeRF MLP.
    sigma : (N_rays, N_samples) torch.Tensor
        Predicted densities (units: 1 / world-space distance).
    t     : (N_rays, N_samples) torch.Tensor
        Sample depths **along each ray** (monotonically increasing).

    Returns
    -------
    dict
        ``"rgb"``   -> (N_rays, 3)   rendered colour  
        ``"depth"`` -> (N_rays,)     expected depth ∑w_i t_i  
        ``"acc"``   -> (N_rays,)     accumulated opacity ∑w_i  
        ``"weights"`` -> (N_rays, N_samples)  per-sample weights (for hierarchical sampling).

    Notes
    -----
    * `1e10` padding on the last Δ behaves like an “infinite” far depth so the
      final α captures any residual σ.  
    * `+1e-10` inside the cumprod avoids log-domain underflow when all α≈1.  
    * Returned ``weights`` should sum to ≤1 (background weight = 1-acc). Use
      ``weights.detach()`` as the PDF for hierarchical sampling.
    """

    # 1. Compute \delta distances between adjacent samples along ray
    delta = t[..., 1:] - t[..., :-1]                 # (N_rays, N_samples-1)
    delta = torch.cat([delta, 1e10*torch.ones_like(delta[..., :1])], -1)  # pad last

    # 2. Convert raw density σ to alpha
    alpha = 1.0 - torch.exp(-sigma * delta)          # (N_rays, N_samples)

    # 3. Compute transmittance T_i = Π_{j<i} (1-α_j)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 
                                1.0 - alpha + 1e-10], -1), -1)[..., :-1]

    # 4. Weight = T * α           (N_rays, N_samples)
    weights = T * alpha

    # 5. Outputs
    rgb_map   = (weights[..., None] * rgb).sum(dim=-2)   # (N_rays, 3)
    depth_map = (weights * t).sum(dim=-1)                # (N_rays)
    acc_map   = weights.sum(dim=-1)                      # (N_rays)
    
    return {
        "rgb":   rgb_map,
        "depth": depth_map,
        "acc":   acc_map,
        "weights": weights          # needed for hierarchical sampling
    }



def hierarchical_sampling(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_importance: int, 
    rand: bool=True 
    ) -> torch.Tensor:
    """
    Fine importance sampling along rays (NeRF “hierarchical sampling”).

    A second, *fine* set of sample depths is drawn from the probability
    distribution implied by the coarse pass's weights.  Mathematically:

        -  Convert the discrete weights w_i to a PDF over bin segments
           (add ε to avoid zeros), then to the corresponding CDF.
        -  Draw `n_importance` uniform samples u ∈ (0,1).
        -  Invert the CDF with piece-wise linear interpolation
           - depth values t_fine that concentrate where coarse weights were high.

    Parameters
    ----------
    bins : (N_rays, N_bins) torch.Tensor
        Mid-points or edges of the coarse depth bins (same ordering as weights).
    weights : (N_rays, N_bins) torch.Tensor
        Weights returned by `volume_rendering()` for each coarse sample.  They
        approximate the true visibility * probability of radiance along the ray.
    n_importance : int
        Number of fine samples to draw per ray.
    rand : bool, default=True
        • True  - jittered (Monte-Carlo) sampling.
        • False - deterministic mid-points in each of `n_importance` equal CDF
          segments (never hits exactly 0 or 1).

    Returns
    -------
    t_fine : (N_rays, n_importance) torch.Tensor
        Fine sample depths suitable for concatenation with the original `bins`
        (then sorted) before feeding through the fine MLP.

    Notes
    -----
    *   `cdf` length is `N_bins+1` because a 0 is prepended; when gathering from
        `bins`, indices must therefore be clamped to `[0, N_bins-1]`, whereas
        gathering from `cdf` may use `[0, N_bins]`.
    *   Adding 1e-5 to `weights` prevents divisions by zero when a ray's coarse
        weights sum to ~0 (e.g., background rays).
    *   All operations are vectorised across rays - no Python loops.
    """
    N_rays, N_bins = weights.shape

    # PDF -> CDF
    pdf = weights + 1e-5                               # prevent n/0
    pdf = pdf / torch.sum(pdf, dim=-1, keepdim=True)   # (N_rays,N_bins)
    cdf = torch.cumsum(pdf, dim=-1)                    # (N_rays,N_bins)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # prepend 0

    # Draw uniform samples u
    if rand:
        u = torch.rand(N_rays, n_importance, device=bins.device)
    else: # mid-points avoid hitting 0 or 1 exactly
        u = (torch.arange(n_importance, device=bins.device) + 0.5) / n_importance
        u = u.expand(N_rays, n_importance)


    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0 )
    above = torch.clamp(inds,     max=N_bins)
    above_bins = torch.clamp(inds,     max=N_bins-1)

    cdf_g0  = torch.gather(cdf,  -1, below)
    cdf_g1  = torch.gather(cdf,  -1, above)
    bins_g0 = torch.gather(bins, -1, below)
    bins_g1 = torch.gather(bins, -1, above_bins)

    denom = (cdf_g1 - cdf_g0)
    denom[denom < 1e-5] = 1.0 # numerical saftey 
    t_fine = bins_g0 + (u - cdf_g0) / denom * (bins_g1 - bins_g0)   # (N_rays,n_importance)
    return t_fine

def pipeline_wrapper(
    ray_o: torch.Tensor,                        # (N_rays, 3)  world-space origins
    ray_d: torch.Tensor,                        # (N_rays, 3)  world-space directions (unit-norm)
    model_coarse: nn.Module,                    # coarse NeRF MLP
    model_fine:   nn.Module,                    # fine   NeRF MLP (can share weights)
    n_coarse: int,                              # # stratified samples per ray
    n_fine:   int,                              # # importance samples per ray
    near: float,                                # near-plane distance
    far:  float,                                # far-plane  distance
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Forward MLP pass that renders a batch of rays through coarse & fine NeRF models.
    Stitches togetherr two sampling stages.

    --> Runs per ray-batch

    Args:
        - ray_o         : torch.Tensor, (N_rays, 3)  world-space origins
        - ray_d         : torch.Tensor, (N_rays, 3)  world-space directions (unit-norm)
        - model_coarse  : nn.Module, coarse NeRF MLP
        - model_fine    : nn.Module, fine NeRF MLP (can share weights)
        - n_coarse      : int, stratified samples per ray
        - n_fine        : int, importance samples per ray
        - near          : float, near-plane distance
        - far           :  float, far-plane  distance
        - encode_fn     : Callable[[torch.Tensor], torch.Tensor]

    Returns
    -------
    {
        "coarse": { "rgb": (N_rays,3), "depth": (N_rays,), "acc": (N_rays,),
                    "weights": (N_rays, n_coarse) },
        "fine":   { "rgb": (N_rays,3), "depth": (N_rays,), "acc": (N_rays,),
                    "weights": (N_rays, n_coarse + n_fine) }
    }
    """
    
    # 1. Stratified sampling
    t_coarse = stratified_samples(ray_d, n_coarse, near, far)    # (N_rays, n_coarse)
    pts_c = ray_o[..., None, :] + ray_d[..., None, :] * t_coarse[..., :, None]

    # 2. Positional encode & MLP
    h_c = encode_fn(pts_c)                                        # (N_rays,n_c,enc_dim)
    rgb_c, sigma_c = model_coarse(h_c, ray_d)                     # predict
    out_c = volume_rendering(rgb_c, sigma_c, t_coarse)            # dict

    # 3. Hierarchical sampling around high-weight bins
    t_fine = hierarchical_sampling(t_coarse, out_c["weights"],
                                   n_fine, rand=True)            # (N_rays,n_fine)
    # concat & sort
    t_all = torch.sort(torch.cat([t_coarse, t_fine], -1), -1).values
    pts_f = ray_o[..., None, :] + ray_d[..., None, :] * t_all[..., :, None]

    # 4. Fine MLP
    h_f = encode_fn(pts_f)
    rgb_f, sigma_f = model_fine(h_f, ray_d)
    out_f = volume_rendering(rgb_f, sigma_f, t_all)

    # 5. Return everything needed for loss & logging
    return {"coarse": out_c, "fine": out_f}



def generate_rays(
    height: int,
    width: int,
    focal: float,
    c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct per-pixel ray origins and directions in world coordinates
    for a pinhole camera that follows the NeRF convention:

        - Camera space axes: +X right, +Y up, -Z forward  
        - Principal point: centre of the image (W / 2, H / 2)  
        - Square pixels: focal length identical in x and y (fx = fy = focal)

    --> Runs per-image

    Method
    ------
    1. Build pixel-centre grids (i,j).  
    2. Convert to camera-space directions  
       d_cam = ((i - W/2) / f, -(j - H/2) / f, -1).  
    3. Rotate directions into world space: d_world = d_cam · R^{T},  
       where R = `c2w[:3, :3]`.  
    4. Normalise every direction.  
    5. Broadcast the camera centre o_world = `c2w[:3, 3]` as the origin
       for all rays.

    Parameters
    ----------
    height : int
        Image height **H** in pixels.
    width : int
        Image width **W** in pixels.
    focal : float
        Focal length in pixels (fx = fy = focal).
    c2w : np.ndarray, shape (4, 4)
        Homogeneous camera-to-world transform.  
        - `c2w[:3, :3]` -> rotation R  
        - `c2w[:3,  3]` -> translation t

    Returns
    -------
    rays_o : np.ndarray, shape (H, W, 3)
        Ray origins in world space (all equal to t).
    rays_d : np.ndarray, shape (H, W, 3)
        Unit-length ray directions in world space.

    Notes
    -----
    * The function intentionally uses NumPy for easy integration with
      image-space preprocessing; convert to torch (`torch.from_numpy`) after
      batching if training on GPU.
    """


    # Create a grid of pixel coordinates 
    i, j = np.meshgrid( np.arange( width ), np.arange( height ), indexing='xy' )

    # Convert image space --> camera space 
    x_cam = (i - width/2) / focal 
    y_cam = -(j - height/2) / focal 
    z_cam = -np.ones_like(i)

    # Stack into direciton vector
    negative_one = -1 # standard convention: camera looks down -Z axis in camera space
    dirs = np.stack( [x_cam, y_cam, z_cam], negative_one)  # shape (H, W, 3)

    # Rotate ray directions from camera to world
    rays_d = dirs @ c2w[ :3, :3 ].T # Rotate ray directions, this is where the ray starts!
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)  # normalize

    # Ray origins are the same for all pixels  - the camera center
    rays_o = np.broadcast_to( c2w[:3, 3], rays_d.shape )

    return rays_o, rays_d # Bothh shapes should be (H, W, 3)
