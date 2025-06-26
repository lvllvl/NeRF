import torch
import numpy as np

def positional_encoding(
    x: torch.Tensor,
    num_freqs: int,
    include_input: bool = True,
    log_sampling: bool = True
) -> torch.Tensor:

    ''' 
    Apply NeRF positional encoding to an N-D tensor of positions or directions.

    Parameters
    ----------
    x               : torch.Tensor
                      Arbitrary shape (..., D). D= 3 for xyz; D = 3 for view dirs.
    num_freqs       : int
                      Number of frequency bands L. Each band adds sin and cos, so output
                      dimension = D * (include_input + 2 * L)
    
    include_input   : bool, default True
                     - True: frequencies = 2^k for k in [0, L-1]  
                     - False: frequencies = 1, 2, ..., L

    Returns
    -------
    torch.Tensor
        Same leading shape as x, last-dim = D * (include_input + 2 * L). 
    ''' 
    if log_sampling:

        freq_bands = 2.0 ** torch.arange( num_freqs, device = x.device, dtype = x.dtype )
    else:
        freq_bands = torch.linspace( 1.0, float( num_freqs ), num_freqs,
                                        device = x.device, dtype = x.dtype )

    # Shape to broadcast: (..., 1, D) * (L, 1) → (..., L, D)
    xb = x.unsqueeze(-2) * freq_bands.unsqueeze(-1) * torch.pi  # (..., L, D)

    encodings = []
    if include_input:
        encodings.append(x)

    # sin / cos  →  (..., L, D) → (..., L*D)
    sin_feat = torch.sin(xb).reshape(*x.shape[:-1], -1)         # flatten L
    cos_feat = torch.cos(xb).reshape(*x.shape[:-1], -1)
    encodings.extend([sin_feat, cos_feat])                      # all rank-k

    return torch.cat( encodings, dim= -1 )