import torch
import os

def unpack_pt_file(pt_file_path, output_dir=None):
    """
    Unpack and inspect the contents of a .pt file.

    Args:
        pt_file_path (str): Path to the .pt file.
        output_dir (str, optional): Directory to save unpacked contents. 
                                    If None, contents are only printed.
    """
    # Check if the file exists
    if not os.path.isfile(pt_file_path):
        print(f"Error: File '{pt_file_path}' does not exist.")
        return

    # Load the .pt file
    try:
        data = torch.load(pt_file_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Inspect the contents
    print(f"Loaded data type: {type(data)}")
    print(data[0][0])
    print(data[1][0])
    print(type(data[2][0]))
    print(type(data[3][0]))
    print(type(data[4][0]))
    print(data[5][0])



    
    # # If the data is a dictionary, print the keys
    # if isinstance(data, dict):
    #     print("Contents of the file:")
    #     for key in data.keys():
    #         print(f"  - {key}: {type(data[key])}, shape: {getattr(data[key], 'shape', 'N/A')}")
    # else:
    #     print("File contains:", type(data), getattr(data, 'shape', 'N/A'))

# Example Usage
# Replace 'example.pt' with the path to your .pt file
# Replace 'output_directory' with a directory where you want to save unpacked components
pt_file_path = "/scratch/alexis/data/0_26158_0.pt"

unpack_pt_file(pt_file_path)
