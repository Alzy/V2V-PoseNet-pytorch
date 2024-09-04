import os
import numpy as np
import torch
from torch.utils.data import Dataset


class HEATDataset(Dataset):
    def __init__(self, base_dir, voxel_suffix, joint_suffix, should_augment=True):
        """
        Args:
            base_dir (string): Directory containing the base files and the 'augmented_dir' subdirectory.
            voxel_suffix (string): Suffix for voxel files (e.g., 'vrt_hand' or 'vrt_body').
            joint_suffix (string): Suffix for joint files (e.g., 'crd_hand' or 'crd_body').
        """
        self.num_joints = 0
        self.base_dir = base_dir
        self.voxel_suffix = voxel_suffix
        self.joint_suffix = joint_suffix
        self.augmented_dir = os.path.join(base_dir, 'augmented')
        self.should_augment = should_augment

        self.base_files = sorted([f for f in os.listdir(base_dir) if f.endswith(f'.{voxel_suffix}.vox.bin')])

        # Collect augmented files if the directory exists
        if os.path.exists(self.augmented_dir):
            self.augmented_files = sorted([f for f in os.listdir(self.augmented_dir) if f.endswith(f'.{voxel_suffix}.vox.bin')])
        else:
            self.augmented_files = []

        # Combine both sets of files
        self.all_files = self.base_files + self.augmented_files

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        """
        Returns next item from the dataset.
        Return values are the input voxel grid and joint heatmaps.
        """
        if idx < len(self.base_files):
            # Base files
            voxel_path = os.path.join(self.base_dir, self.all_files[idx])
            joint_path = voxel_path.replace(f'.{self.voxel_suffix}.vox.bin', f'.{self.joint_suffix}.bin')
        else:
            # Augmented files
            voxel_path = os.path.join(self.augmented_dir, self.all_files[idx])
            joint_path = voxel_path.replace(f'.{self.voxel_suffix}.vox.bin', f'.{self.joint_suffix}.bin')

        voxel_grid = np.fromfile(voxel_path, dtype=np.uint32).astype(np.float32).reshape((88, 88, 88)).copy()
        joint_locations = np.fromfile(joint_path, dtype=np.float32).reshape((-1, 3)).copy()

        assert voxel_grid.shape == (88, 88, 88), f"Unexpected voxel_grid shape: {voxel_grid.shape}"
        assert joint_locations.shape[1] == 3, f"Unexpected joint_locations shape: {joint_locations.shape}"
        assert self.num_joints > 0

        # Apply augmentation
        if self.should_augment:
            voxel_grid, joint_locations = self.augment(voxel_grid, joint_locations)

        # Generate joint location heat maps
        heat_maps = self._generate_heatmaps(joint_locations, (88, 88, 88))

        # Downsample the heatmaps by a factor of 2 (pool_factor)
        downsampled_heat_maps = np.zeros((self.num_joints, 44, 44, 44))
        for i in range(self.num_joints):
            downsampled_heat_maps[i] = heat_maps[i, ::2, ::2, ::2]

        # Add the channel dimension to voxel_grid
        voxel_grid = voxel_grid.reshape((1, *voxel_grid.shape))

        return torch.from_numpy(voxel_grid), torch.from_numpy(downsampled_heat_maps)

    def _generate_heatmaps(self, joint_coordinates, voxel_grid_size, sigma=0.11):
        """
        Generate heatmaps for a set of joints.

        Args:
            joint_coordinates (list): List of normalized joint coordinates (x, y, z).
            voxel_grid_size (tuple): Size of the heatmap (D, H, W).
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            numpy.ndarray: Array of heatmaps for each joint.
        """
        # self.num_joints = len(joint_coordinates)
        heatmaps = np.zeros((self.num_joints, *voxel_grid_size), dtype=np.float32)

        for i, joint in enumerate(joint_coordinates):
            heatmaps[i] = self._create_gaussian_heatmap(voxel_grid_size, joint, sigma)

        return heatmaps

    def _create_gaussian_heatmap(self, voxel_grid_size, center, sigma):
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

    def augment(self, voxel_grid, joint_locations):
        """
        Augment the data. This method should be overridden by derived classes.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
