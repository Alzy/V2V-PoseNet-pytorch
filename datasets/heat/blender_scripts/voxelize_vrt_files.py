import os
import subprocess


def run_voxelizer(directory):
    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".vrt_hand.bin"):
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
    # Replace these with your directory paths
    directory_paths = [
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand1',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand2',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand3',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand4',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand5',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand6',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand7',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand8',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand9',
        'Z:\\Jeffs corner\\hand datasets\\hands\\2f_rd3\\hand10',
        # Add more paths as needed
    ]

    for path in directory_paths:
        run_voxelizer(path)
