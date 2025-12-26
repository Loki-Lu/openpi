import h5py

def print_hdf5_tree(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")

h5_path = "/gemini-2/user/private/data/organize_small/hdf5/0.hdf5"

with h5py.File(h5_path, "r") as f:
    f.visititems(print_hdf5_tree)