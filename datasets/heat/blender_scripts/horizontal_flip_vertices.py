from mathutils import Vector
import numpy as np


def horizontal_flip_vectors(vrt_file_path, crd_file_path):
    """
    Performs a horizontal flip on the x-axis for both vertex vectors and joint vectors.

    Args:
        vrt_file_path: File containing list of vertex coordinates.
        crd_file_path: File containing list of normalized joint coordinates.

    Returns:
        (list of numpy f32, list of numpy f32): Flipped vert_values and joint_values.
    """

    # Read the values from the file
    values = np.fromfile(vrt_file_path, dtype=np.float32)
    joint_coords = np.fromfile(crd_file_path, dtype=np.float32)

    min_bounds = values[:3]
    max_bounds = values[3:6]
    _vert_values = values[6:]

    vert_values = [Vector(_vert_values[i:i+3]) for i in range(0, len(_vert_values), 3)]
    joint_values = [Vector(joint_coords[i:i+3]) for i in range(0, len(joint_coords), 3)]

    min_bounds = Vector(min_bounds)
    max_bounds = Vector(max_bounds)

    # Calculate the midpoint along the x-axis
    mid_x = (min_bounds.x + max_bounds.x) / 2

    # Flip vertex values
    flipped_vert_values = [Vector((2 * mid_x - v.x, v.y, v.z)) for v in vert_values]

    # Flip joint values
    flipped_joint_values = [Vector((1 - v.x, v.y, v.z)) for v in joint_values]

    # Transform to numpy arrays
    flipped_vert_values = np.array([component for vec in flipped_vert_values for component in vec], dtype=np.float32)
    flipped_joint_values = np.array([component for vec in flipped_joint_values for component in vec], dtype=np.float32)

    return flipped_vert_values, flipped_joint_values


def horizontal_flip_voxel_grid(vox_file_path):
    """
    Performs a horizontal flip on the x-axis for the voxel grid.

    Args:
        vox_file_path: File containing the voxel grid as a 1D float32 array.

    Returns:
        np.ndarray: Flipped voxel grid as a 1D float32 array.
    """

    # Read the voxel grid from the file
    voxel_grid = np.fromfile(vox_file_path, dtype=np.float32)
    # Reshape the 1D array to a 3D array (88x88x88)
    voxel_grid = voxel_grid.reshape((88, 88, 88))

    # Flip the voxel grid along the x-axis
    flipped_voxel_grid = np.flip(voxel_grid, axis=2)

    # Reshape the 3D array back to a 1D array
    flipped_voxel_grid = flipped_voxel_grid.flatten().astype(np.float32)

    return flipped_voxel_grid


# Example usage
if __name__ == "__main__":
    _vrt_file_path = "D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.vrt_hand.bin"
    _vox_file_path = "D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.vrt_hand.vox.bin"
    _crd_file_path = "D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.crd_hand.bin"
    _flipped_vert_values, _flipped_joint_values = horizontal_flip_vectors(_vrt_file_path, _crd_file_path)
    _flipped_vox_values = horizontal_flip_voxel_grid(_vox_file_path)

    # Save transformed values
    # _flipped_vert_values.tofile("D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.f.vrt_hand.bin")
    _flipped_vox_values.tofile("D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.f.vrt_hand.vox.bin")
    _flipped_joint_values.tofile("D:\\HEAT Dataset\processed\\0c01f7cf47444e428f4c64126b9bf642.f.crd_hand.bin")
