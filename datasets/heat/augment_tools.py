import numpy as np
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R


def rotate_voxel_and_joints(voxel_grid, joint_coords, angle, axis='y'):
    """
    Rotate both voxel grid and joint coordinates.

    :param voxel_grid: 3D numpy array representing the voxel grid
    :param joint_coords: 2D numpy array of joint coordinates (N x 3)
    :param angle: Rotation angle in degrees
    :param axis: Rotation axis ('x', 'y', or 'z')
    :param pivot_point: Pivot point for rotation (default is center)
    :return: Rotated voxel grid and rotated joint coordinates
    """
    # Define axis mappings for voxel grid rotation (SciPy convention)
    voxel_axis_mappings = {
        'x': (0, 1),
        'y': (2, 0),
        'z': (1, 2)
    }

    # Define axis mappings for joint coordinate rotation (THREE.js convention)
    joint_axis_mappings = {
        'x': ('x', -1),
        'y': ('y', 1),
        'z': ('z', -1)
    }

    # Voxel grid rotation
    rotated_grid = rotate(voxel_grid, angle=angle, axes=voxel_axis_mappings[axis], reshape=False, order=1)

    # Joint coordinate rotation
    pivot_point = np.array([0.5, 0.5, 0.5])

    # Center joint coordinates around pivot point
    joint_coords_centered = joint_coords - pivot_point

    # Create rotation object
    joint_rotation_axis, angle_multiplier = joint_axis_mappings[axis]
    rotation = R.from_euler(joint_rotation_axis, angle * angle_multiplier, degrees=True)

    # Apply rotation to joint coordinates
    rotated_joint_coords_centered = rotation.apply(joint_coords_centered)

    # Move joint coordinates back to original position
    rotated_joint_coords = rotated_joint_coords_centered + pivot_point

    return rotated_grid, rotated_joint_coords


if __name__ == "__main__":
    # File paths
    voxel_file_path = r"D:\0.vrt_body.vox.bin"
    coord_file_path = r"D:\0.crd_body.bin"

    # Grid dimensions
    grid_dimensions = (88, 88, 88)

    # Read voxel grid
    with open(voxel_file_path, "rb") as f:
        voxel_grid_1d = np.fromfile(f, dtype=np.uint32)
    voxel_grid = voxel_grid_1d.reshape(grid_dimensions)

    # Read joint coordinates
    with open(coord_file_path, "rb") as f:
        joint_coords = np.fromfile(f, dtype=np.float32).reshape(-1, 3)

    # Example rotations
    axes = ['x', 'y', 'z']
    for axis in axes:
        print(f"\nRotating around {axis}-axis:")
        rotated_grid, rotated_joint_coords = rotate_voxel_and_joints(voxel_grid, joint_coords, angle=45, axis=axis)

        # Save rotated voxel grid
        with open(f"D:\\0_rotated_{axis}.vrt_body.vox.bin", "wb") as f:
            rotated_grid.astype(np.uint32).flatten().tofile(f)

        # Save rotated joint coordinates
        with open(f"D:\\0_rotated_{axis}.crd_body.bin", "wb") as f:
            rotated_joint_coords.astype(np.float32).flatten().tofile(f)

        print(f"Rotation complete. Rotated files saved with '{axis}' prefix.")
        print("First few original coordinates:", joint_coords[:3])
        print("First few transformed coordinates:", rotated_joint_coords[:3])
