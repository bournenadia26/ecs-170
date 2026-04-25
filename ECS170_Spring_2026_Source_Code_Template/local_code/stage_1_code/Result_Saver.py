'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        import os
        print('saving results...')
        # Use os.path.join for robust path construction
        filename = self.result_destination_file_name + '_' + str(self.fold_count)
        full_path = os.path.join(self.result_destination_folder_path, filename)
        print(f"[DEBUG] Saving result to: {full_path}")
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        with open(full_path, 'wb') as f:
            pickle.dump(self.data, f)