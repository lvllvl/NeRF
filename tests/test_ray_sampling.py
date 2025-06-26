import torch, math 
from utils.ray_sampling import stratified_samples 

def test_stratified_shapes():

    N = 4
    ray_d = torch.randn( N, 3 )

# def stratified_samples(
#     ray_d: torch.Tensor,
#     num_samples: int,
#     near: float,
#     far: float,
#     rand: bool = True, 
# ) -> torch.Tensor:
    t = stratified_samples( ray_d=ray_d, num_samples=8, near=0.5, far=2.0, rand=False )
    assert t.shape == ( N, 8 )

    # deterministic branch should be stricktly increasing
    assert torch.all( t[:, 1:] > t[:, :-1] )