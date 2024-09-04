import numpy as np
from scipy.ndimage import rotate
from scipy.spatial.transform import Rotation as R

# Define the file paths
voxel_file_path = r"D:\0.vrt_hand.vox.bin"
coord_file_path = r"D:\0.crd_hand.bin"
output_voxel_file_path = r"D:\0_rotated.vrt_hand.vox.bin"
output_coord_file_path = r"D:\0_rotated.crd_hand.bin"

# Define the grid dimensions
grid_dimensions = (88, 88, 88)

# Read the 1D uint32 array from the voxel file
with open(voxel_file_path, "rb") as f:
    voxel_grid_1d = np.fromfile(f, dtype=np.uint32)

# Reshape the 1D array into a 3D voxel grid
voxel_grid = voxel_grid_1d.reshape(grid_dimensions)

# Rotate the grid around the center by 45 degrees
rotated_grid = rotate(voxel_grid, angle=45, axes=(1, 0), reshape=False, order=1)

# Flatten the rotated grid back into a 1D array
rotated_grid_1d = rotated_grid.flatten()

# Write the rotated 1D array to a new binary file
with open(output_voxel_file_path, "wb") as f:
    rotated_grid_1d.astype(np.uint32).tofile(f)

# Read the 1D f32 array from the coordinate file
with open(coord_file_path, "rb") as f:
    joint_coords = np.fromfile(f, dtype=np.float32)

# Reshape the 1D array into a set of 3D coordinates
joint_coords = joint_coords.reshape(-1, 3)

# Define the center of the joint coordinate space
center = np.array([0.5, 0.5, 0.5])

# Translate coordinates to be centered around the origin
joint_coords_centered = joint_coords - center

# Create a rotation object for 45 degrees around the y-axis
rotation = R.from_euler('x', -45, degrees=True)

# Apply the rotation to the joint coordinates
rotated_joint_coords_centered = rotation.apply(joint_coords_centered)

# Translate the coordinates back to their original position
rotated_joint_coords = rotated_joint_coords_centered + center

# Flatten the rotated coordinates back into a 1D array
rotated_joint_coords_1d = rotated_joint_coords.flatten()

# Write the rotated 1D array to a new binary file as float32
with open(output_coord_file_path, "wb") as f:
    rotated_joint_coords_1d.astype(np.float32).tofile(f)

print(f"Rotation complete. Rotated voxel grid saved to {output_voxel_file_path}")
print(f"Rotated joint coordinates saved to {output_coord_file_path}")

# Print the original and transformed coordinates for comparison
print("Original coordinates:")
print(','.join(map(str, joint_coords.flatten())))
print("\nTransformed coordinates:")
print(','.join(map(str, rotated_joint_coords_1d)))