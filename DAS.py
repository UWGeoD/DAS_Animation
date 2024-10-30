import numpy as np
import h5py
from scipy import signal
from queue import Queue
import Utilities

class DAS:
    def __init__(self, file, select_channels=None) -> None:
        self.file = file
        self.data = None
        self.meta_data = dict()
        self.meta_data['time'] = str( Utilities.parse_datetime_from_filename(self.file) )
        self.select_channels = select_channels
        self._get_data()

    # To do: need to be able to read different type of das files
    def _get_data(self):
        with h5py.File(self.file, "r") as fp:
            data = fp["Acquisition/Raw[0]/RawData"][...]
            if self.select_channels is not None:
                self.data = self._select_channels(data.T, self.select_channels)
            else:
                self.data = data.T
            self.meta_data['dx'] = fp["Acquisition"].attrs["SpatialSamplingInterval"]
            self.meta_data['fs'] = fs = fp['Acquisition/Raw[0]'].attrs["OutputDataRate"]
            self.meta_data['dt'] = 1.0 / fs
    def plot(self, start_time=None, end_time=None, title=None):
        Utilities.plot_das_data(process_data(self.data), self.select_channels, self.meta_data['dx'], self.meta_data['dt'], 
                      start_time, end_time, self.meta_data['time'])
    def _select_channels(self, data, channels):
        return data[channels, :]
        
# need a red-black tree for insert and order new file(date, file)
# rb tree (date), dict (date: file), time complexity O(log n)
# or can sort the dict directly, time complexity O(n log n)
# think: 
# 1. separate _get_meta_data from _get_data, or could be an parameter of _get_data
# 2. a normal get_data function outside of the DAS class (could make the class easier to read)
# 3. Given a MulDAS obj. How to find next file? (in order to do a waterfall movie) 
# Should have a file list in the begining, and process files in chunk way.
# 4. Consider not reading all files
class MulDAS(DAS):
    def __init__(self, file_list, select_channels=None) -> None:
        self.file_list = file_list
        self.time_file = self._create_time_file_dict()
        self.select_channels = select_channels
        self._get_data()
        #self.meta_data['time'] = f"{parse_datetime_from_filename(self.file_list[0])[0:13]}:00"
    def _get_data(self):
        data_combined = None
        i = 0 
        #### ORDER 
        for t, file in self.time_file.items():
            if self.select_channels is not None:
                DAS_temp = DAS(file, self.select_channels)
            else:
                DAS_temp = DAS(file)
            if i == 0:
                data_combined = DAS_temp.data
                self.meta_data = DAS_temp.meta_data
            else:
                data_temp = DAS_temp.data
                data_combined = np.concatenate([data_combined, data_temp], axis=1)
            i += 1
        self.data = data_combined

    def _create_time_file_dict(self):
        return Utilities.create_time_file_dict(self.file_list)
        
    def append(self, file_list):
        pass



#class MulDAS2(DAS):



# queue element(DAS obj) with stride parameter 



# should put this to Utilities.py and add more flexibility
# Check what DASpy do for preprocessing
def process_data(data):
    data_detrend = signal.detrend(data, type='linear')
    sos = signal.butter(5, [20,100], 'bandpass', fs=500, output='sos')
    data_filtered = signal.sosfilt(sos, data_detrend)
    return data_filtered

