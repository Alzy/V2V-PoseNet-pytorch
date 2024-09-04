import numpy as np
from heat_dataset import HEATDataset


def horizontal_flip_data(voxel_grid, joint_coordinates):
    """
    Horizontally flip voxel grid data and corresponding joint coordinates to simulate opposite hand data
    """
    assert joint_coordinates.shape[1] == 3, "Joint locations should have shape (-1, 3)"

    # Flip the x-axis coordinates
    flipped_joint_coordinates = joint_coordinates.copy()
    flipped_joint_coordinates[:, 0] = 1 - joint_coordinates[:, 0]

    # Flip the voxel grid along the x-axis
    flipped_voxel_grid = np.flip(voxel_grid, axis=2).copy()

    return flipped_voxel_grid, flipped_joint_coordinates


class HEATHandDataset(HEATDataset):
    def __init__(self, base_dir):
        super().__init__(base_dir, voxel_suffix='vrt_hand', joint_suffix='crd_hand')
        self.num_joints = 16  # Specify the number of joints for the hand dataset

    def augment(self, voxel_grid, joint_locations):
        """
        Perform hand-specific augmentation, such as horizontal flipping.
        """
        if np.random.rand() < 0.5:
            return horizontal_flip_data(voxel_grid, joint_locations)

        return voxel_grid, joint_locations