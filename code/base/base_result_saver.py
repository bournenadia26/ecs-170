'''
Base Result Saver class for ECS 170 Stage 3
'''

class Result_Saver:
    def __init__(self, sName=None, sDescription=None):
        self.sName = sName
        self.sDescription = sDescription
        self.result_destination_folder_path = None
        self.result_destination_file_name = None

    def save(self, raw_result, dataset_name=''):
        raise NotImplementedError('save() must be implemented by subclass')
