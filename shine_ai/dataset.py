from sklearn.model_selection import train_test_split

import numpy as np
class DataSet():
    def __init__(self, name, parser, experiment_index, test_percentage, random_state=93):
        self.name = name
        self.parser = parser
        self.random_state = random_state
        self.test_percentage = test_percentage
        self.experiment_index = experiment_index

        self.samples, self.shine_through = parser.data(self.experiment_index)
        self.remove_nans()

        self.x = np.array(list(self.samples.values())).T
        self.minimum, self.maximum = np.min(self.x.T, axis=1).reshape(-1,1), np.max(self.x.T, axis=1).reshape(-1,1)
        
        self.x_norm = self.normalise(self.x)
        self.n_samples = len(self.x)
        
        self.split()

    
    def remove_nans(self):
        print('REMOVING NANS')
        mask = np.invert(np.isnan(self.shine_through))
        self.samples = {k:v[mask] for k,v in self.samples.items()}
        self.shine_through = self.shine_through[mask]
        print(f'\nN SAMPLES AFTER NANS REMOVED = {len(self.shine_through)}')

    def normalise(self, x):
        x_norm = (x.T-self.minimum)/(self.maximum-self.minimum)
        return x_norm.T

    def split(self):    
        print(f'\nRANDOMLY SPLITTING DATA INTO TEST AND TRAINING SETS: {self.test_percentage}% test, {100-self.test_percentage}% training.')
        self.x_train, self.x_test, self.shine_train, self.shine_test = train_test_split(self.x, self.shine_through, test_size=self.test_percentage/100, random_state=self.random_state)
        self.x_train_norm, self.x_test_norm = self.normalise(self.x_train), self.normalise(self.x_test)