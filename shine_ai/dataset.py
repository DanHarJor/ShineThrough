import numpy as np
class DataSet():
    def __init__(self, name, parser, experiment_index):
        self.name = name
        self.parser = parser

        self.experiment_index = experiment_index

        self.samples, self.shine_through = parser.data(self.experiment_index)
        self.x = self.x_norm()
        
    def x_norm(self):
        # Max Min normalisation is used on the data but not the labels
        values = np.array(list(self.samples.values()))
        minimum, maximum = np.min(values, axis=1).reshape(-1,1), np.max(values, axis=1).reshape(-1,1)
        # minimum, maximum = np.repeat(minimum.T, values.shape[1], axis=1), np.repeat(maximum.T, values.shape[1], axis=1)
        values_norm = (values-minimum)/(maximum-minimum)
        x_norm = values_norm.T
        return x_norm