'''
Base Method class for ECS 170 Stage 3
'''

class Method:
    def __init__(self, mName=None, mDescription=None):
        self.mName = mName
        self.mDescription = mDescription

    def fit(self, X, y):
        raise NotImplementedError('fit() must be implemented by subclass')

    def predict(self, X):
        raise NotImplementedError('predict() must be implemented by subclass')

    def run(self):
        raise NotImplementedError('run() must be implemented by subclass')
