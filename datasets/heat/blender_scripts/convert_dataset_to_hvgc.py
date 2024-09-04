import os
import zipfile
import numpy as np


def compress_voxel_and_coordinates(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".vrt_body.vox.bin"):
            base_filename = filename.replace(".vrt_body.vox.bin", "")
            vox_file = os.path.join(directory, base_filename + ".vrt_body.vox.bin")
            crd_file = os.path.join(directory, base_filename + ".crd_body.bin")

            if os.path.exists(crd_file):
                # Load the data from the files
                voxel_data = np.fromfile(vox_file, dtype=np.uint32)
                coordinate_data = np.fromfile(crd_file, dtype=np.float32)

                # Create the output filename for the zip file
                output_filename = os.path.join(directory, base_filename + ".zip")

                # Create a zip archive and add both files
                with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add the voxel grid data
                    zipf.writestr(base_filename + ".vrt_body.vox.bin", voxel_data.tobytes())
                    # Add the coordinate data
                    zipf.writestr(base_filename + ".crd_body.bin", coordinate_data.tobytes())

                print(f"Compressed {vox_file} and {crd_file} into {output_filename}")
            else:
                print(f"Coordinate file {crd_file} not found, skipping {vox_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python convert_dataset_to_hvgc.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        sys.exit(1)

    compress_voxel_and_coordinates(directory)
