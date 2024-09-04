import numpy as np
import argparse
import os


def compress_voxel_grid(voxel_grid: np.ndarray, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Compress a 1D numpy array representing a voxel grid using run-length encoding (RLE).

    The compression algorithm iterates through the voxel grid array, counting the number
    of consecutive occurrences of each value and storing them as pairs (value, count).
    If the count exceeds the maximum representable value for the specified dtype (255 for
    uint8, 4294967295 for uint32), it splits the count into multiple entries.

    Args:
        voxel_grid (np.ndarray): A 1D numpy array representing the voxel grid.
        dtype (np.dtype, optional): The target dtype for the compressed array. Default is uint8.

    Returns:
        np.ndarray: A compressed numpy array using the specified dtype.
    """
    # Determine the max value based on the dtype
    max_value = np.iinfo(dtype).max

    # Initialize an empty list to store the compressed values
    compressed_values = []

    # Start iterating over the voxel grid array
    i = 0
    while i < len(voxel_grid):
        # Get the current value and initialize the occurrence counter
        current_value = voxel_grid[i]
        count = 1

        # Count consecutive occurrences of the current value
        while i + 1 < len(voxel_grid) and voxel_grid[i + 1] == current_value:
            count += 1
            i += 1

        # Split counts greater than max_value into multiple entries
        while count > max_value:
            compressed_values.append(current_value)
            compressed_values.append(max_value)
            count -= max_value

        # Store the value and its count in the compressed values list
        compressed_values.append(current_value)
        compressed_values.append(count)

        # Move to the next distinct value in the array
        i += 1

    # Convert the compressed values list to a numpy array with the specified dtype
    return np.array(compressed_values, dtype=dtype)


def decompress_voxel_grid(compressed_grid: np.ndarray, dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Decompress a numpy array that was compressed using run-length encoding (RLE) back to its
    original 1D voxel grid representation.

    The method iterates through the compressed array, which contains pairs of values and their
    respective counts, and reconstructs the original array by repeating each value according to
    its count.

    Args:
        compressed_grid (np.ndarray): A compressed numpy array using run-length encoding.
        dtype (np.dtype, optional): The target dtype for the decompressed array. Default is uint8.

    Returns:
        np.ndarray: The original, uncompressed 1D voxel grid array.
    """
    # Initialize an empty list to store the decompressed values
    decompressed_values = []

    # Iterate over the compressed array in steps of 2 (value, count pairs)
    for i in range(0, len(compressed_grid), 2):
        value = compressed_grid[i]       # The voxel value
        count = compressed_grid[i + 1]   # The number of occurrences

        # Repeat the value 'count' times and extend the decompressed values list
        decompressed_values.extend([value] * count)

    # Convert the decompressed values list to a numpy array
    return np.array(decompressed_values, dtype=dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress or decompress a voxel grid using run-length encoding.")
    parser.add_argument("filename", type=str, help="The input file name (for both compression and decompression).")
    parser.add_argument("--dtype", type=str, choices=["uint8", "uint32"], default="uint8", help="The dtype to use for compression/decompression. Default is uint8.")
    parser.add_argument("--mode", type=str, choices=["zip", "unzip"], default="zip", help="Mode: 'zip' to compress, 'unzip' to decompress. Default is 'zip'.")
    parser.add_argument("--delete-original", action="store_true", help="Delete the original file after compression/decompression.")

    args = parser.parse_args()

    # Load the file
    dtype = np.uint8 if args.dtype == "uint8" else np.uint32
    voxel_grid = np.fromfile(args.filename, dtype=dtype)

    if args.mode == "zip":
        compressed_grid = compress_voxel_grid(voxel_grid, dtype=dtype)
        output_filename = os.path.splitext(args.filename)[0] + ".voxzip"
        compressed_grid.tofile(output_filename)
        print(f"Compressed file saved as {output_filename}")
    else:  # args.mode == "unzip"
        decompressed_grid = decompress_voxel_grid(voxel_grid, dtype=dtype)
        output_filename = os.path.splitext(args.filename)[0] + ".vox.bin"
        decompressed_grid.tofile(output_filename)
        print(f"Decompressed file saved as {output_filename}")

    # Delete the original file if the flag is set
    if args.delete_original:
        os.remove(args.filename)
        print(f"Original file {args.filename} deleted.")