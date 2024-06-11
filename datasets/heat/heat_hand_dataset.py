import os
import numpy as np
import sys
import struct
from torch.utils.data import Dataset


def _create_gaussian_heatmap(voxel_grid_size, center, sigma):
    """
    Create a 3D Gaussian heatmap centered at a specific point.

    Args:
        voxel_grid_size (tuple): Size of the heatmap (D, H, W).
        center (tuple): Center of the Gaussian (normalized coordinates between 0 and 1).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: 3D heatmap.
    """
    D, H, W = voxel_grid_size
    z, y, x = np.meshgrid(np.linspace(0, 1, D), np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
    z = (z - center[2]) ** 2
    y = (y - center[1]) ** 2
    x = (x - center[0]) ** 2
    heatmap = np.exp(-(x + y + z) / (2 * sigma ** 2))
    return heatmap


def horizontal_flip_data(voxel_grid, joint_coordinates):
    """
    Horizontally flip voxel grid data and corresponding joint coordinates to simulate opposite hand data
    """
    assert joint_coordinates.shape[1] == 3, "Joint locations should have shape (-1, 3)"

    # Flip the x-axis coordinates
    flipped_joint_coordinates = joint_coordinates.copy()
    flipped_joint_coordinates[:, 0] = 1 - joint_coordinates[:, 0]

    # Flip the voxel grid along the x-axis
    voxel_grid = np.flip(voxel_grid, axis=2)

    return voxel_grid, joint_coordinates


def _generate_heatmaps(joint_coordinates, voxel_grid_size, sigma=0.05):
    """
    Generate heatmaps for a set of joints.

    Args:
        joint_coordinates (list): List of normalized joint coordinates (x, y, z).
        voxel_grid_size (tuple): Size of the heatmap (D, H, W).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: Array of heatmaps for each joint.
    """
    num_joints = len(joint_coordinates)
    heatmaps = np.zeros((num_joints, *voxel_grid_size), dtype=np.float32)

    for i, joint in enumerate(joint_coordinates):
        heatmaps[i] = _create_gaussian_heatmap(voxel_grid_size, joint, sigma)

    return heatmaps


class HEATHandDataset(Dataset):
    def __init__(self, base_dir):
        """
        Args:
            base_dir (string): Directory containing the base files and the 'augmented_dir' subdirectory.
        """
        self.base_dir = base_dir
        self.augmented_dir = os.path.join(base_dir, 'augmented')

        self.base_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.vox.bin')])

        # Collect augmented files if the directory exists
        if os.path.exists(self.augmented_dir):
            self.augmented_files = sorted([f for f in os.listdir(self.augmented_dir) if f.endswith('.vox.bin')])
        else:
            self.augmented_files = []

        # Combine both sets of files
        self.all_files = self.base_files + self.augmented_files

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if idx < len(self.base_files):
            # Base files
            voxel_path = os.path.join(self.base_dir, self.all_files[idx])
            joint_path = voxel_path.replace('.vox.bin', '.crd_hand.bin')
        else:
            # Augmented files
            voxel_path = os.path.join(self.augmented_dir, self.all_files[idx])
            joint_path = voxel_path.replace('.vox.bin', '.crd_hand.bin')

        voxel_grid = np.fromfile(voxel_path, dtype=np.float32).reshape((88, 88, 88))
        joint_locations = np.fromfile(joint_path, dtype=np.float32).reshape((-1, 3))
        if np.random.rand() < 0.5:
            # horizontal flip half the time to simulate opposite hand
            voxel_grid, joint_locations = horizontal_flip_data(voxel_grid, joint_locations)

        heat_maps = _generate_heatmaps(joint_locations, (88, 88, 88))

        return {'voxel_grid': voxel_grid, 'heat_maps': heat_maps}
