import json
import math
import os
from pathlib import Path
from PIL import Image
import pytest

def find_image_file(base_path: str, valid_extensions=(".png", ".jpg", ".jpeg")) -> str:
    # If base_path already ends in a known extension, use directly
    if os.path.exists(base_path):
        return base_path

    # Otherwise, try common extensions
    for ext in valid_extensions:
        candidate = base_path + ext
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"No image file found for base path: {base_path}")

def validate_image_dimensions(transforms_path: str, images_dir: str, expected_width: int, expected_height: int):
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)

    for frame in transforms["frames"]:
        raw_file_path = os.path.join(images_dir, frame["file_path"])
        img_path = find_image_file(raw_file_path)

        with Image.open(img_path) as img:
            width, height = img.size

        if (width, height) != (expected_width, expected_height):
            raise ValueError(f"{img_path} has size {(width, height)} but expected {(expected_width, expected_height)}")

def test_validate_image_dimensions(tmp_path):
    # Setup dummy image and transform
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    image_path = img_dir / "test_0000.png"
    img = Image.new("RGB", (800, 600))  # 800x600 dummy image
    img.save(image_path)

    # Setup transforms.json
    transforms = {
        "camera_angle_x": 0.6911,
        "frames": [
            {"file_path": "images/test_0000"}
        ]
    }

    json_path = tmp_path / "transforms.json"
    with open(json_path, "w") as f:
        json.dump(transforms, f)

    # Run test
    validate_image_dimensions(json_path, tmp_path, expected_width=800, expected_height=600)


def test_real_dataset_image_shapes():
    
    # path to raw dataset, transforms.json 
    dataset_dir = Path("data/")
    transforms_json_path =  "data/colmap_output/transforms.json"
    
    # Expected image resolution
    expected_width = 3213 
    expected_height = 5712 

    # Raise error if any image is missing, mis-sized
    validate_image_dimensions(
        transforms_path=transforms_json_path,
        images_dir=dataset_dir,
        expected_width=expected_width,
        expected_height=expected_height
    )