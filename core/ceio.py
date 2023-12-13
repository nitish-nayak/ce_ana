import numpy as np
import h5py
from measure import *

# serialize a measurement object to a hdf5 file
class serialize:
    def __init__(self, meas_data, filename):
        self.save_h5py(meas_data, filename)

    @classmethod
    def save_h5py(cls, meas_data, filename):
        with h5py.File(filename, 'a') as h5file:
            if isinstance(meas_data, (measurement, waveform)):
                cls.serialize_dict({meas_data.cfg : meas_data.meas}, h5file, '/'+meas_data.femb+'/'+meas_data.test+'/')
            elif isinstance(meas_data, dict):
                cls.serialize_dict(meas_data, h5file, '/')
            else:
                raise NotImplementedError("Cannot save measurement of type %s" % type(item))

    @classmethod
    def serialize_dict(cls, data_dic, h5file, path):
        for key, item in data_dic.items():
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
                h5file[path+key] = item
            elif isinstance(item, dict):
                cls.serialize_dict(item, h5file, path+key+'/')
            elif isinstance(item, (measurement, waveform)):
                cls.serialize_meas(item.meas, h5file, path+key+'/')
            else:
                raise NotImplementedError("Cannot save measurement contents of type %s" % type(item))

    @classmethod
    def serialize_meas(cls, meas, h5file, path):
        if isinstance(meas, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path] = meas
        else:
            raise NotImplementedError("Cannot save measurement contents of type %s" % type(item))
