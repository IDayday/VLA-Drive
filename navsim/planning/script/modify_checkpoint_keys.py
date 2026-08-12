import torch
import argparse
import os
from collections import OrderedDict


def modify_checkpoint_keys(checkpoint_path: str):
    """
    Loads a PyTorch checkpoint, renames keys in the state_dict, and saves a new checkpoint.
    Keys containing "pad_model" will be renamed to "drivor_model".

    :param checkpoint_path: Path to the input checkpoint file.
    """
    # Check if the file exists
    if not os.path.isfile(checkpoint_path):
        print(f"Error: File not found at {checkpoint_path}")
        return

    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Use 'cpu' map_location to avoid GPU memory usage
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # The state_dict could be at the top level or inside a 'state_dict' key (for PyTorch Lightning)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Create a new state_dict with modified keys
    new_state_dict = OrderedDict()
    keys_changed_count = 0
    for key, value in state_dict.items():
        if "pad_model" in key:
            new_key = key.replace("pad_model", "drivor_model")
            new_state_dict[new_key] = value
            print(f"  - Renamed key: '{key}' -> '{new_key}'")
            keys_changed_count += 1
        else:
            new_state_dict[key] = value

    if keys_changed_count == 0:
        print("No keys containing 'pad_model' were found. No new file will be created.")
        return

    print(f"\nTotal keys renamed: {keys_changed_count}")

    # Update the checkpoint with the new state_dict
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
    else:
        checkpoint = new_state_dict

    # Define the new file path
    dir_name, base_name = os.path.split(checkpoint_path)
    new_file_name = f"new_{base_name}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # Save the new checkpoint
    print(f"Saving new checkpoint to {new_file_path}...")
    torch.save(checkpoint, new_file_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modify keys in a PyTorch checkpoint by replacing "pad_model" with "drivor_model".')
    parser.add_argument('checkpoint_path', type=str, help='Path to the .pth or .ckpt checkpoint file.')
    args = parser.parse_args()
    modify_checkpoint_keys(args.checkpoint_path)
