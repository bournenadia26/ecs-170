'''
Base Evaluate Metrics class for ECS 170 Stage 3
'''

class Evaluate_Metrics:
    def __init__(self, eName=None, eDescription=None):
        self.eName = eName
        self.eDescription = eDescription

    def evaluate(self, pred_y, true_y):
        raise NotImplementedError('evaluate() must be implemented by subclass')
