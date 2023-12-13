import numpy as np

# measurement class
class measurement:
    def __init__(self, meas, fembid=None, test=None, meastype=np.float64, cfg=None):
        self.meas = meas
        self.femb = fembid
        self.test = test
        self.meastype = meastype
        self.cfg = cfg
    def setConfiguration(self, cfg):
        self.cfg = cfg
    def setFEMB(self, fembid):
        self.femb = fembid
    def setTest(self, test):
        self.test = test

# waveform class
class waveform(measurement):
    def __init__(self, meas, fembid=0, test=None, cfg=None):
        measurement.__init__(self, meas, fembid=fembid, test=test, meastype=np.ndarray, cfg=cfg)
    def nChannels(self):
        return self.meas.shape[0]
    def nTicks(self):
        return self.meas.shape[1]

# define a wrapper class for functions of measurements
class var:
    def __init__(self, fn, vartype=np.float64):
        self.fn = fn
        self.vartype = vartype
    def __call__(self, imeas, *args, **kwargs):
        val = self.vartype(self.fn(imeas.meas, **kwargs), *args)
        ret = measurement(val, fembid=imeas.femb, test=imeas.test, meastype=self.vartype, cfg = imeas.cfg)

        return ret
