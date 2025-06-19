from PIL import Image
import os

image_dir = "data/raw"

# load one sample 
sample_image = os.path.join( image_dir, "IMG_1033.jpeg")
with Image.open(sample_image) as img:
    width, height = img.size
    print( f"Sample image size: {width} x {height}")