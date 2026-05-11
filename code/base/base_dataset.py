'''
Base Dataset Loader class for ECS 170 Stage 3
'''

class Dataset_Loader:
    def __init__(self, seed=None, dName=None):
        self.seed = seed
        self.dName = dName
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load(self):
        raise NotImplementedError('load() must be implemented by subclass')
