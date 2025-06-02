import h5py

h5_path = '/Volumes/T7 Shield/dT_0.1K_200K_3500K_compressed/capture_xs_data_0.h5'

def print_h5_structure(name, obj):
    """
    Callback for h5py.File.visititems to print each object’s name,
    type, shape (if a dataset), and dtype (if a dataset).
    """
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}/")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}  shape={obj.shape}  dtype={obj.dtype}")

with h5py.File(h5_path, "r") as f:
    print(f"Keys at root: {list(f.keys())}\n")
    f.visititems(print_h5_structure)
