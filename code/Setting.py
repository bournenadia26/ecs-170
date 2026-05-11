'''
Setting class for ECS 170 Stage 3
Wires together: Dataset_Loader, Method (CNN), Result_Saver, Evaluate_Metrics
'''

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Setting:
    def __init__(self, sName=None, sDescription=None):
        self.sName = sName
        self.sDescription = sDescription

        self.dataset        = None   # Dataset_Loader instance
        self.method         = None   # Method_CNN instance
        self.result_saver   = None   # Result_Saver instance
        self.evaluate       = None   # Evaluate_Metrics instance
        self.device         = None   # torch.device

    def load_run_save_evaluate(self):
        import torch

        # --- 1. Determine device ---
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        print(f'\n[Setting] Using device: {self.device}')

        # --- 2. Load data ---
        train_loader, test_loader = self.dataset.load()

        # --- 3. Move model to device ---
        self.method.to(self.device)

        # --- 4. Train ---
        print(f'[Setting] Training {self.method.mName} ...')
        loss_history, acc_history = self.method.train_model(train_loader, self.device)

        # --- 5. Test ---
        print(f'[Setting] Testing {self.method.mName} ...')
        pred_y, true_y = self.method.test_model(test_loader, self.device)

        # --- 6. Evaluate ---
        metrics = self.evaluate.print_report(pred_y, true_y, dataset_name=self.dataset.dName)

        # --- 7. Save results ---
        raw_result = {
            'loss_history': loss_history,
            'acc_history':  acc_history,
            'metrics':      metrics,
            'pred_y':       pred_y,
            'true_y':       true_y,
        }
        self.result_saver.save(raw_result, dataset_name=self.dataset.dName)

        return metrics
