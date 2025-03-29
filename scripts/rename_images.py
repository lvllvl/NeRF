import os

def rename_images( folder_path, prefix="plant_", ext=".jpg" ):

    images = sorted( [f for f in os.listdir( folder_path ) if f.lower().endswith(".jpeg" )])
    for idx, filename in enumerate( images, start=1 ):
        new_name = f"{prefix}{idx:04d}{ext}"
        src = os.path.join( folder_path, filename )
        dst = os.path.join( folder_path, new_name )
        os.rename( src, dst )

    print( f"Renamed {len(images)} images." )

if __name__ == "__main__":
    rename_images( "data/raw/plant/images" )
