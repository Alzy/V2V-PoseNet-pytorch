import os
import numpy as np


def check_crd_files(directory):
    invalid_file_count = 0
    # Loop through all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".crd_hand.bin"):
            abs_path = os.path.join(directory, filename)
            base_path = abs_path.replace('.crd_hand.bin', '')

            # Read the values from the file
            values = np.fromfile(abs_path, dtype=np.float32)

            # Check for values greater than 1 or less than 0
            if np.any(values > 1.01) or np.any(values < -0.01):
                print(f"File {abs_path} contains values out of range (0, 1).")
                vrt_file = base_path + '.vrt_hand.bin'
                vox_file = base_path + '.vrt_hand.vox.bin'
                # delete files
                os.remove(vrt_file)
                os.remove(vox_file)
                os.remove(abs_path)
                invalid_file_count += 1

    print(f"Found {invalid_file_count} invalid CRD files")


if __name__ == "__main__":
    # Replace 'your_directory_path_here' with the path to your directory
    directory_path = 'D:\\HEAT Dataset\\processed\\augmented'
    check_crd_files(directory_path)
