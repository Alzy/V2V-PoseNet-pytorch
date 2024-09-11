import numpy as np
from .augment_tools import rotate_voxel_and_joints
from .heat_dataset import HEATDataset


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

        initial_hand_location = joint_locations[0]

        hand_location = None

        if initial_hand_location[2] > 0.33 and initial_hand_location[2] < 0.66:
            if initial_hand_location[0] < 0.5 and initial_hand_location[1] < 0.6:
                hand_location = 'left'
            elif initial_hand_location[0] < 0.5 and initial_hand_location[1] > 0.6:
                hand_location = 'corner'
            elif initial_hand_location[0] > 0.25 and initial_hand_location[1] > 0.6:
                hand_location = 'top'

        x_angle = 0
        y_angle = 0
        z_angle = 0

        if hand_location == 'left':
            x_angle = np.random.uniform(-90, 90)
            y_angle = np.random.uniform(-15, 15)
            z_angle = np.random.uniform(0, 30)
        if hand_location == 'corner':
            x_angle = np.random.uniform(-90, 90)
            y_angle = np.random.uniform(-15, 15)
            z_angle = np.random.uniform(0, 30)
        if hand_location == 'top':
            x_angle = np.random.uniform(-15, 15)
            y_angle = np.random.uniform(-90, 90)
            z_angle = np.random.uniform(-30, 0)

        if hand_location and np.random.rand() < 0.5:
            voxel_grid, joint_locations = rotate_voxel_and_joints(voxel_grid, joint_locations, angle=x_angle, axis='x')
            voxel_grid, joint_locations = rotate_voxel_and_joints(voxel_grid, joint_locations, angle=y_angle, axis='y')
            voxel_grid, joint_locations = rotate_voxel_and_joints(voxel_grid, joint_locations, angle=z_angle, axis='z')

        if np.random.rand() < 0.5:
            return horizontal_flip_data(voxel_grid, joint_locations)

        return voxel_grid, joint_locations