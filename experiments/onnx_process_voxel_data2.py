import numpy as np
import matplotlib.pyplot as plt
import os
import onnxruntime as ort
import torch

from src.v2v_util import extract_coord_from_output


def load_voxel_data(filename):
    """Loads voxel grid data from a .bin file into the shape (1, 1, 88, 88, 88)."""
    voxel_grid = np.fromfile(filename, dtype=np.uint32).astype(np.float32).reshape((88, 88, 88))
    print(f"Voxel grid loaded from {filename}:")
    print(f"Max value in grid: {voxel_grid.max()}")

    # Add the batch dimension to match the expected shape (1, 1, 88, 88, 88)
    voxel_grid = voxel_grid[np.newaxis, np.newaxis, :, :, :]
    print('Voxel grid shape after adding batch and channel dimensions:', voxel_grid.shape)

    return voxel_grid


def load_voxel_txt_data(filename):
    """Loads voxel grid data from a text file into the shape (1, 88, 88, 88)."""
    with open(filename, 'r') as file:
        # Read all lines at once
        lines = file.read().strip().split('\n')

        # Assuming each line is a separate slice of 88x88
        # Initialize a list to hold each 2D slice array
        slices = []

        # Process each line/slice
        for line in lines:
            if line.strip():  # Ensure the line isn't empty
                if len(line) == 0: continue
                # Convert line (a string of numbers separated by commas) to a numpy array
                slice_array = np.fromstring(line, sep=', ', dtype=float).reshape(88, 88, 88)
                slices.append(slice_array)

        # Stack slices to form a 3D array with dimensions (88, 88, 88)
        voxel_grid = np.stack(slices, axis=0)

        print('voxel grid shape:', voxel_grid.shape)  # Shape is (1, 88, 88, 88) at this point
        # Add the batch dimension to match the original shape (1, 88, 88, 88)
        voxel_grid = voxel_grid[np.newaxis, :, :, :]  # Shape becomes (1, 1, 88, 88, 88)
        print('voxel grid shape:', voxel_grid.shape)
        voxel_grid = voxel_grid.astype(np.float32)

    return voxel_grid


def run_model(session, voxel_data):
    # Prepare the input to the model as a dictionary
    # The input name 'input' should be the name of the input tensor in the ONNX model
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: voxel_data})
    return outputs[0]


def transform_output(heatmaps):
    """
    Transforms the output heatmaps back to normalized coordinates.

    Args:
        heatmaps (numpy.ndarray): Array of heatmaps (batch, num_joints, D, H, W).

    Returns:
        numpy.ndarray: Normalized coordinates of keypoints (batch, num_joints, 3).
    """
    # Ensure heatmaps are in the correct shape
    batch_size, num_joints, D, H, W = heatmaps.shape

    # Initialize an array to hold the keypoints
    keypoints = np.zeros((batch_size, num_joints, 3), dtype=np.float32)

    # Loop through each sample in the batch
    for b in range(batch_size):
        for j in range(num_joints):
            # Get the heatmap for the current joint
            heatmap = heatmaps[b, j]

            # Find the index of the maximum value in the heatmap
            z, y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

            # Convert the coordinates to the normalized range [0, 1]
            keypoints[b, j, 0] = x / W
            keypoints[b, j, 1] = y / H
            keypoints[b, j, 2] = z / D

    return keypoints


def visualize_heatmaps(heatmaps, num_heatmaps=5):
    """
    Visualize the first few heatmaps returned by the model.

    Args:
        heatmaps (numpy.ndarray or torch.Tensor): The heatmaps to visualize. Shape (batch, joints, depth, height, width) or (joints, depth, height, width).
        num_heatmaps (int): Number of heatmaps to visualize. Default is 5.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().detach().numpy()

    print(f"Heatmaps shape before squeezing: {heatmaps.shape}")  # Debug statement to print the shape of the heatmaps

    # Remove the batch dimension if present
    if len(heatmaps.shape) == 5:
        heatmaps = np.squeeze(heatmaps, axis=0)

    print(f"Heatmaps shape after squeezing: {heatmaps.shape}")  # Debug statement to print the shape of the heatmaps
    assert len(heatmaps.shape) == 4, f"Expected heatmaps to have 4 dimensions, but got {heatmaps.shape}"

    num_heatmaps = min(num_heatmaps, heatmaps.shape[0])

    fig, axes = plt.subplots(num_heatmaps, 3, figsize=(15, 5 * num_heatmaps))

    for i in range(num_heatmaps):
        # Take the middle slice of each dimension for visualization
        middle_slice = heatmaps[i, heatmaps.shape[1] // 2, :, :]
        height_slice = heatmaps[i, :, heatmaps.shape[2] // 2, :]
        width_slice = heatmaps[i, :, :, heatmaps.shape[3] // 2]

        if num_heatmaps == 1:
            axes[0].imshow(middle_slice, cmap='hot', interpolation='nearest')
            axes[0].set_title(f'Joint {i} - Depth Slice')
            axes[1].imshow(height_slice, cmap='hot', interpolation='nearest')
            axes[1].set_title(f'Joint {i} - Height Slice')
            axes[2].imshow(width_slice, cmap='hot', interpolation='nearest')
            axes[2].set_title(f'Joint {i} - Width Slice')
        else:
            axes[i, 0].imshow(middle_slice, cmap='hot', interpolation='nearest')
            axes[i, 0].set_title(f'Joint {i} - Depth Slice')
            axes[i, 1].imshow(height_slice, cmap='hot', interpolation='nearest')
            axes[i, 1].set_title(f'Joint {i} - Height Slice')
            axes[i, 2].imshow(width_slice, cmap='hot', interpolation='nearest')
            axes[i, 2].set_title(f'Joint {i} - Width Slice')

    plt.tight_layout()
    plt.show()


def main(voxel_file, model_dir, epoch_number):
    # Load ONNX model
    model_path = os.path.join(model_dir, f'epoch{epoch_number}.onnx')
    session = ort.InferenceSession(model_path)

    # Load voxel data
    # voxel_data = load_voxel_txt_data(voxel_file)
    voxel_data = load_voxel_data(voxel_file)

    # Run model
    output = run_model(session, voxel_data)
    print("Output shape:", output.shape)
    # keypoints = extract_coord_from_output(output)
    keypoints = transform_output(output)
    print("Model output:", keypoints)
    print("Keypoints shape:", keypoints.shape)
    visualize_heatmaps(output)

    keypoints.flatten().tofile("D:\\testing.crd_hand.bin")


if __name__ == "__main__":
    # voxel_file = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\outputs\voxel_grid_0.txt'
    voxel_file = "D:\\HEAT Dataset\\processed\\33ed6f5047144af5ad6b059858c57ccf.vrt_hand.vox.bin"
    model_dir = 'C:\\Users\\gonza\\Desktop\\V2V-PoseNet-pytorch\\onnx'
    epoch = 23

    main(voxel_file, model_dir, epoch)
