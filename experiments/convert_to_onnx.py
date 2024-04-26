import torch
import argparse
import os
from torch.autograd import Variable
from src.v2v_model import V2VModel  # Ensure this is the correct path to your model's definition

def load_checkpoint(model, checkpoint_path):
    """
    Load weights from the checkpoint into a PyTorch model.
    This function assumes that the checkpoint is a dictionary that contains
    the model's state under the key 'model_state_dict' among other metadata.
    """
    # Load the entire checkpoint dictionary
    checkpoint = torch.load(checkpoint_path)

    # Check if the expected key 'model_state_dict' exists
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the expected key does not exist, raise an error
        raise KeyError("No model state found in the checkpoint file.")

    return model

def export_to_onnx(model, input_size, output_file):
    """
    Export the given model to an ONNX file.
    """
    # Create a dummy variable with the correct input size
    dummy_input = Variable(torch.randn(*input_size))

    # Export the model
    torch.onnx.export(model, dummy_input, output_file, verbose=True)

def main(epoch, checkpoint_dir='../checkpoint', onnx_dir='../onnx'):
    """
    Main function to load a checkpoint and export it to ONNX.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(script_dir, checkpoint_dir)
    onnx_dir = os.path.join(script_dir, onnx_dir)

    checkpoint_file = os.path.join(checkpoint_dir, f'epoch{epoch}.pth')
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"No checkpoint file found for epoch {epoch} at {checkpoint_file}")

    # Ensure the output directory exists
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)

    # Initialize model
    model = V2VModel(input_channels=1, output_channels=21)  # Adjust based on your model specification
    model = load_checkpoint(model, checkpoint_file)
    model.eval()  # Set the model to inference mode

    # Input Tensor Dimensions:
    # Batch Size (N): Represents the number of examples in the batch. When creating a dummy input for model export, you typically use a batch size of 1 for simplicity.
    # Channels (C): Typically, the channel dimension for a voxel grid might be 1 (if the grid contains only occupancy information) unless the model is designed to take multiple types of information or derived features per voxel.
    # Depth (D): The size of the voxel grid along the depth axis.
    # Height (H): The size of the voxel grid along the vertical axis.
    # Width (W): The size of the voxel grid along the horizontal axis
    input_size = (1, 1, 88, 88, 88)

    # Set the path for the ONNX file
    output_file = os.path.join(onnx_dir, f'epoch{epoch}.onnx')

    # Export the model
    export_to_onnx(model, input_size, output_file)
    print(f"Model from epoch {epoch} exported to ONNX format at {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PyTorch model checkpoint to ONNX format')
    parser.add_argument('epoch', type=int, help='Epoch number of the model to convert')
    args = parser.parse_args()

    main(args.epoch)
