from PIL import Image
import os

def png_to_gif(input_folder, output_gif, duration=500, loop=0):
    """
    Convert a sequence of PNG images in a folder into a GIF animation.

    :param input_folder: Path to the folder containing PNG images
    :param output_gif: Path to save the output GIF file
    :param duration: Duration of each frame in milliseconds (default: 500ms)
    :param loop: Number of times the GIF should loop (0 means infinite)
    """
    # Get a sorted list of PNG files in the folder
    png_files = sorted([file for file in os.listdir(input_folder) if file.lower().endswith('.png')])

    if not png_files:
        print("No PNG files found in the input folder!")
        return

    frames = []
    for png_file in png_files:
        # Open each PNG image and append it to the frames list
        frame_path = os.path.join(input_folder, png_file)
        try:
            frame = Image.open(frame_path)
            frames.append(frame)
        except Exception as e:
            print(f"Error loading {png_file}: {e}")

    if len(frames) > 1:
        # Save the frames as a GIF
        frames[0].save(
            output_gif,
            format='GIF',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop
        )
        print(f"GIF saved at {output_gif}")
    else:
        print("Not enough frames to create an animated GIF.")

# Example usage

input_folder = "./data/my_imgs/"  # 替换为包含PNG文件的文件夹路径
output_gif = "./data/res_imgs/1"      # 只需提供文件名，程序会自动加上 .gif
png_to_gif(input_folder, output_gif, duration=300, loop=0)
