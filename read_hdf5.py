import h5py

def print_hdf5_contents(file_name):
    def print_attrs(name, obj):
        print(f"{name}:")
        for key, val in obj.attrs.items():
            print(f"    {key}: {val}")

    with h5py.File(file_name, 'r') as f:
        f.visititems(print_attrs)

# Example usage
file_name = 'my-data/master.h5'
print_hdf5_contents(file_name)
