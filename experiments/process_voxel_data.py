import torch
import numpy as np
import os
from src.v2v_model import V2VModel
from src.v2v_util import extract_coord_from_output


def load_voxel_data(filename):
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
                print('.')
                # Convert line (a string of numbers separated by commas) to a numpy array
                slice_array = np.fromstring(line, sep=', ', dtype=float).reshape(88, 88, 88)
                slices.append(slice_array)

        # Stack slices to form a 3D array with dimensions (88, 88, 88)
        voxel_grid = np.stack(slices, axis=0)

        print('voxel grid shape:', voxel_grid.shape)  # Shape is (1, 88, 88, 88) at this point
        # Add the batch dimension to match the original shape (1, 88, 88, 88)
        voxel_grid = voxel_grid[np.newaxis, :, :, :]  # Shape becomes (1, 1, 88, 88, 88)
        print('voxel grid shape:', voxel_grid.shape)

    return voxel_grid

def load_checkpoint(model, checkpoint_path):
    """Loads model weights from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

def run_model(model, voxel_data):
    """Runs the model on the voxel data."""
    model.eval()
    with torch.no_grad():
        data_tensor = torch.from_numpy(voxel_data).to(torch.float32)
        output = model(data_tensor)
    return output

def main(voxel_file, checkpoint_dir, epoch_number):
    # Model and checkpoint configuration
    input_channels = 1  # Adjust if different
    output_channels = 21  # Adjust based on your model output configuration

    # Load model
    print("Loading model...")
    model = V2VModel(input_channels, output_channels)
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch{epoch_number}.pth')
    load_checkpoint(model, checkpoint_path)

    # Load voxel data
    print("Loading voxel data..")
    voxel_data = load_voxel_data(voxel_file)

    # Run model
    output = run_model(model, voxel_data)
    keypoints = extract_coord_from_output(output)
    print("Model output:", keypoints)

if __name__ == "__main__":
    voxel_file = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\outputs\voxel_grid_0.txt'
    checkpoint_dir = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\checkpoint'
    epoch = 14

    main(voxel_file, checkpoint_dir, epoch)
