import os

class Config:
    def __init__(self):
        self.RESULTS_DIR = 'csv_files'
        self.PLOTS_DIR = 'plots'

    def get_results_dir(self):
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        return self.RESULTS_DIR

    def get_plots_dir(self):
        os.makedirs(self.PLOTS_DIR, exist_ok=True)
        return self.PLOTS_DIR

    def get_results_path(self, filename):
        return os.path.join(self.RESULTS_DIR, filename)

    def get_plots_path(self, filename):
        '''
        Get the full path for a plot file in the plots directory.

        Args:
            filename (str): Name of the plot file (e.g., "robust_accuracy.png")
        '''
        return os.path.join(self.PLOTS_DIR, filename)
