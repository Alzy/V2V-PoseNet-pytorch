import os
import subprocess


def run_voxelizer(directory):
    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".vrt_body.bin"):
            abs_path = os.path.join(directory, filename)
            base_name = os.path.splitext(os.path.basename(abs_path))[0]
            output_path = os.path.join(directory, f"{base_name}.vox.bin")
            command = f"HEATVoxelizerWGPU.exe \"{abs_path}\" \"{output_path}\""

            # Skip if the output file already exists
            if os.path.exists(output_path):
                print(f"Skipping {abs_path}, output file already exists.")
                continue

            # Execute the command
            subprocess.run(command, shell=True)
            print(f"Processed {abs_path}")


if __name__ == "__main__":
    # Replace 'your_directory_path_here' with the path to your directory
    directory_path = 'D:\HEAT Dataset\processed_body\\augmented'
    run_voxelizer(directory_path)
