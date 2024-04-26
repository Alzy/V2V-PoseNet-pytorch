import numpy as np
import os
from datasets.msra_hand import MARAHandDataset
from src.v2v_util import V2VVoxelization

data_dir = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\datasets\msra_hand'
center_dir = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\datasets\msra_center'
output_dir = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\outputs'

def save_voxel_grid_to_txt(voxel_grid, filename):
    """ Saves the voxel grid to a text file, handling 3D arrays by saving each 2D slice separately. """
    with open(filename, 'w') as f:
        print('saving...', filename)
        num_slices = voxel_grid.shape[0]
        print(voxel_grid.shape)
        for i in range(num_slices):
            # Write each slice to the file, add a newline between slices for clarity
            voxel_grid.tofile(f, sep=', ')
            f.write("\n")


def main(data_dir, center_dir, output_dir):
    # Configuration
    keypoints_num = 21
    cubic_size = 200
    test_subject_id = 3  # Change as required for different subjects

    # Initialize voxelization utility
    voxelization_util = V2VVoxelization(cubic_size=cubic_size, augmentation=False)

    # Load dataset
    dataset = MARAHandDataset(data_dir, center_dir, 'train', test_subject_id, transform=None)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each depth map in the dataset
    for idx in range(len(dataset)):
        sample = dataset[idx]
        points = sample['points']
        refpoint = sample['refpoint']

        # Voxelization
        voxel_grid = voxelization_util.voxelize(points, refpoint)

        # Save the voxel grid to a text file
        filename = os.path.join(output_dir, f"voxel_grid_{idx}.txt")
        save_voxel_grid_to_txt(voxel_grid, filename)

        print(f"Saved voxel grid {idx} to {filename}")

main(data_dir, center_dir, output_dir)
