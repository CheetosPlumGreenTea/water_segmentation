import os
from PIL import Image

def convert_jpg_to_png(input_folder, output_folder):
    """
    Convert all .jpg files in the input folder to .png files in the output folder.

    :param input_folder: Path to the folder containing .jpg files
    :param output_folder: Path to the folder where .png files will be saved
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            
            try:
                # Open the .jpg image
                with Image.open(input_path) as img:
                    # Convert and save as .png
                    img.save(output_path, "PNG")
                print(f"Converted {filename} to {output_path}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

# Example usage
input_folder = "./data/imgs/"  # 替换为你的JPG文件夹路径
output_folder = "./data/imgs_png/"  # 替换为你希望保存PNG的文件夹路径

convert_jpg_to_png(input_folder, output_folder)
