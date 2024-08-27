import os.path as path
import scipy.io
import numpy as np
from itertools import product

class Parser():
    def __init__(self, file_path):
        self.mat = scipy.io.loadmat(file_path)
        self.inputs = self.inputs()


        self.experiment_labels = self.experiment_labels()

    def inputs(self):
        inputs_labels = [str(i[0]) for i in self.mat['data'][0][0][0][0][1:]]
        inputs_labels
        inputs = {}
        for i, label in enumerate(inputs_labels, start=1):
            inputs[label] = self.mat['data'][0][0][3][0][i][0]
        inputs['injector_number'] = np.array([False, True]) #false for off axis and True for on axis
        return inputs
    
    def experiment_labels(self):
        return [str(i[0]) for i in (self.mat['data'][0][0][3][0][0][0])]
    
    def data(self, experiment_index):
        shine_through_mat = self.mat['data'][0][0][6][experiment_index]
        inputs_arrays = list(self.inputs.values())
        all_combinations_index = np.array(list(product(*[np.arange(len(i)) for i in inputs_arrays])))
        all_combinations = np.array(list(product(*inputs_arrays)))
        shine_through = []
        for index in all_combinations_index:
            shine_through.append(shine_through_mat[*index])
        
        samples = {}
        for i, key in enumerate(list(self.inputs.keys())):
            samples[key] = all_combinations.T[i]
        
        return samples, shine_through
    

