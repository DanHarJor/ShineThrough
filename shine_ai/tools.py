import pickle
def save_pkl(path, var):
    with open(path, 'wb') as pickle_file:
        pickle.dump(var, pickle_file)

def load_pkl(path):
    with open(path, 'rb') as pickle_file:
        return pickle.load(pickle_file)