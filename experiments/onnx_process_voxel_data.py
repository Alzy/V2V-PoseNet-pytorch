import numpy as np
import os
import onnxruntime as ort
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


def main(voxel_file, model_dir, epoch_number):
    # Load ONNX model
    model_path = os.path.join(model_dir, f'epoch{epoch_number}.onnx')
    session = ort.InferenceSession(model_path)

    # Load voxel data
    voxel_data = load_voxel_data(voxel_file)

    # Run model
    output = run_model(session, voxel_data)
    keypoints = extract_coord_from_output(output)
    print("Model output:", keypoints)


if __name__ == "__main__":
    voxel_file = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\outputs\voxel_grid_0.txt'
    model_dir = r'C:\Users\gonza\Desktop\V2V-PoseNet-pytorch\onnx'
    epoch = 14

    main(voxel_file, model_dir, epoch)
