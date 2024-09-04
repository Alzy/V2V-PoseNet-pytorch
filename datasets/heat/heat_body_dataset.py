from .heat_dataset import HEATDataset
from .augment_tools import rotate_voxel_and_joints
import numpy as np


class HEATBodyDataset(HEATDataset):
    def __init__(self, base_dir):
        super().__init__(base_dir, voxel_suffix='vrt_body', joint_suffix='crd_body')
        self.num_joints = 20  # Specify the number of joints for the body dataset

    def augment(self, voxel_grid, joint_locations):
        """
        Perform body-specific augmentation. No horizontal flipping for body data.
        """
        # Implement any specific augmentations for body data here
        x_angle = np.random.uniform(-180, 180)
        y_angle = np.random.uniform(-180, 180)
        z_angle = np.random.uniform(-180, 180)
        rotated_grid, rotated_joint_coords = rotate_voxel_and_joints(voxel_grid, joint_locations, angle=x_angle, axis='x')
        rotated_grid, rotated_joint_coords = rotate_voxel_and_joints(rotated_grid, rotated_joint_coords, angle=y_angle, axis='y')
        rotated_grid, rotated_joint_coords = rotate_voxel_and_joints(rotated_grid, rotated_joint_coords, angle=z_angle, axis='z')
        return rotated_grid, rotated_joint_coords