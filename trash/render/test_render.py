import torch
from render.render_image import render_image
from models.nerf_model import NeRF
from render.save_image import save_image

def test_render():

    # Load the trained NeRF model
    model = NeRF()
    model.load_state_dict( torch.load('checkpoints/model.pth' ))
    model.eval()

    # Define camera pose and parameters for rendering
    camera_pose = torch.eye( 4 ) # example camera pose (identity matrix)
    image_size = (800, 600)
    near, far = 2.0, 6.0 # Define near and far planes

    # render the image
    with torch.no_grad():
        image = render_image( camera_pose, image_size, model, near, far )

    # Save or display image
    save_image( image, 'output/rendered_image.png' )


if __name__ == '__main__':
    test_render()