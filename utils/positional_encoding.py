import torch
import numpy as np

def positional_encoding( positions, num_freqs ):
    """
    Apply positional encoding to the input positions.

    Parameters:
    ---------

    positions: torch.Tensor
        - The 3D coordinates to encode.
    
    num_freqs: int
        - The number of frequency bands for encoding.

    Returns:
    ---------
    torch.Tensor:
        - The encoded positions. 
    """

    encoding = [ positions ]
    for i in range( num_freqs ):

        for fn in [ torch.sin, torch.cos ]:
            encoding.append( fn( 2.0 ** i * positions ) )
    
    return torch.cat( encoding, dim = -1 )
