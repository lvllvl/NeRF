import torch
from utils.positional_encoding import positional_encoding

def test_posenc_shape():
    xyz = torch.randn(5, 3)      # 5 points
    enc = positional_encoding(xyz, num_freqs=6)
    assert enc.shape[-1] == 3 * (1 + 2*6)      # 39
