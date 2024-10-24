import numpy as np
import h5py
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy import signal

class DAS:
    def __init__(self, file) -> None:
        self.file = file
        self.data = None
        self.meta_data = dict()
        self._get_data()
    def _get_data(self):
        with h5py.File(self.file, "r") as fp:
            data = fp["Acquisition/Raw[0]/RawData"][...]
            self.data = data.T
            self.meta_data['dx'] = fp["Acquisition"].attrs["SpatialSamplingInterval"]
            self.meta_data['fs'] = fs = fp['Acquisition/Raw[0]'].attrs["OutputDataRate"]
            self.meta_data['dt'] = 1.0 / fs
    def plot(self, start_time=None, end_time=None, title=None):
        data_plot = deepcopy(self.data)
        data_plot = process_data(data_plot)
        
        nx, nt = data_plot.shape
        x = np.arange(nx) * self.meta_data['dx']
        t = np.arange(nt) * self.meta_data['dt']
    
        if start_time is not None and end_time is not None:
            time_indices = (t >= start_time) & (t <= end_time)
            data_plot = data_plot[:, time_indices]
            t = t[time_indices]
        
        plt.figure()
        plt.imshow(normalize(data_plot).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto", 
                   extent=[x[0], x[-1], t[-1], t[0]], interpolation="none")
        plt.xlabel("Channel Number")
        plt.ylabel("Time (s)")
        plt.title(title)
        

class MulDAS(DAS):
    def __init__(self, file_list) -> None:
        self.file_list = file_list
        self._get_data()
    def _get_data(self):
        data_combined = None
        i = 0 
        for file in self.file_list:
            if i == 0:
                DAS_temp = DAS(file)
                data_combined = DAS_temp.data
                self.meta_data = DAS_temp.meta_data
            else:
                data_temp = DAS(file).data
                data_combined = np.concatenate([data_combined, data_temp], axis=1)
            i += 1
        self.data = data_combined




# should put this to Utilities.py and add more flexibility
def process_data(data):
    data_detrend = signal.detrend(data, type='linear')
    sos = signal.butter(5, [20,100], 'bandpass', fs=500, output='sos')
    data_filtered = signal.sosfilt(sos, data_detrend)
    return data_filtered


normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / np.std(x, axis=-1, keepdims=True)