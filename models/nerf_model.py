import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.positional_encoding import positional_encoding

class NeRF( nn.Module ):

    def __init__( self, num_freqs=10 ):

        super( NeRF, self ).__init__()
        self.num_freqs = num_freqs

        # Define the MLP layers for NeRF
        self.fc_layers = nn.ModuleList( [nn.Linear( 63, 256 ) for _ in range(8) ]) # Input size depends on positional encoding
        self.output_rgb = nn.Linear( 256, 3 )
        self.output_density = nn.Linear( 256, 1 )

    def forward( self, x ):

        # Positional encoding
        x = positional_encoding( x, self.num_freqs )

        # Forward pass through the MLP
        for fc in self.fc_layers:
            x = F.relu( fc(x) )
        
        rgb = torch.sigmoid( self.output_rgb(x) ) # RGB color
        density = F.relu( self.output_density(x) ) # Density (sigma)

        return rgb, density