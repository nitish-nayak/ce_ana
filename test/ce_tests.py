import os
import sys
import pickle
import numpy as np
import h5py
import re
import statsmodels.api as sm

sys.path.append('../decode')
from spymemory_decode import wib_dec

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

# define multiple measurements based on waveforms
def WaveformAna(fembs, rawdata, getwfs=False, rms_flg=False, cfg=None, test=None):
    wibdata = wib_dec(rawdata,fembs, spy_num=1, cd0cd1sync=False)[0]
    datd = [wibdata[0], wibdata[1],wibdata[2],wibdata[3]][fembs[0]]

    chns =[]
    rmss = []
    peds = []
    pkps = []
    pkns = []
    wfs = []

    ppos0=0
    npos0=0
    ppos64=0
    npos64=0
    for achn in range(len(datd)):
        chndata = datd[achn]
        amax = np.max(chndata[300:-150])
        amin = np.min(chndata[300:-150])
        if achn==0:
            ppos0 = np.where(chndata[300:-150]==amax)[0][0] + 300
            npos0 = np.where(chndata[300:-150]==amin)[0][0] + 300
        if achn==64:
            ppos64 = np.where(chndata[300:-150]==amax)[0][0] + 300
            npos64 = np.where(chndata[300:-150]==amin)[0][0] + 300

        if rms_flg:
            arms = np.std(chndata)
            aped = int(np.mean(chndata))
        else:
            if achn <64:
                arms = np.std(chndata[ppos0-150:ppos0-50])
                aped = int(np.mean(chndata[ppos0-150:ppos0-50]))
                wfs.append(np.array(chndata[ppos0-50:ppos0+150]) - aped)
            else:
                arms = np.std(chndata[ppos64-150:ppos64-50])
                aped = int(np.mean(chndata[ppos64-150:ppos64-50]))
                wfs.append(np.array(chndata[ppos64-50:ppos64+150]) - aped)
        chns.append(achn)
        rmss.append(arms)
        peds.append(aped)
        pkps.append(amax)
        pkns.append(amin)

    feid = 'CE'+str(fembs[0])
    m_chns = measurement(np.array(chns), fembid=feid, meastype=np.ndarray, test=test, cfg=cfg)
    m_rmss = measurement(np.array(rmss), fembid=feid, meastype=np.ndarray, test=test, cfg=cfg)
    m_peds = measurement(np.array(peds), fembid=feid, meastype=np.ndarray, test=test, cfg=cfg)
    m_pkps = measurement(np.array(pkps), fembid=feid, meastype=np.ndarray, test=test, cfg=cfg)
    m_pkns = measurement(np.array(pkns), fembid=feid, meastype=np.ndarray, test=test, cfg=cfg)

    if not getwfs:
        return m_chns, m_rmss, m_peds, m_pkps, m_pkns
    else:
        return m_chns, m_rmss, m_peds, m_pkps, m_pkns, wfs


def CapacitanceAna(fembs, rawdata, cfg=None):
    wibdata = wib_dec(rawdata, fembs, spy_num=1, cd0cd1sync=False)[0]
    chnc = int(re.sub(r'FECHN([0-9]*)_.*$', '\\1', cfg))

    chns = np.arange(chnc, chnc+128, 16)
    pps = []
    for femb in fembs:
        for chn in chns:
            pps.append(np.max(wibdata[femb][chn]))

    return chnc, pps

# define power measurement
def kPower(pwr_dict):
    ret = {}
    for key in pwr_dict:
        l = pwr_dict[key]
        vtype = re.sub('FE[0-9]_(.*$)', '\\1', key)
        ret[vtype] = {'V' : np.float64(l[0]), 'I' : np.float64(l[1]), 'VI' : np.float64(l[2])}

    return ret
kPower = var(kPower, vartype=dict)

def kDACLinearFit(dacs):
    x, y = zip(*dacs)
    error_fit = False
    try:
        results = sm.OLS(y,sm.add_constant(x)).fit()
    except ValueError:
        error_fit = True
    if ( error_fit == False ):
        error_gain = False
        try:
            slope = results.params[2]
        except IndexError:
            slope = 0
            error_gain = True
        try:
            constant = results.params[0]
        except IndexError:
            constant = 0
    else:
        slope = 0
        constant = 0
        error_gain = True

    y_fit = np.array(x)*slope + constant
    delta_y = abs(y - y_fit)
    inl = delta_y / (max(y)-min(y))
    peakinl = max(inl)

    ret = {'Slope': np.float64(slope), 'Intercept': np.float64(constant), 'PeakINL': np.float64(peakinl), 'ErrorGain': np.int64(error_gain)}
    return ret
kDACLinearFit = var(kDACLinearFit, vartype=dict)

# load in our data
fsubdir = "FE_001000001_001000002_001000003_001000004_001000005_001000006_001000007_001000008"
froot = "../data/"
fdir = froot + fsubdir + "/"

def load_data(cetype):
    fp = fdir + cetype + ".bin"
    with open(fp, 'rb') as f:
        data = pickle.load(f)

    data.pop("logs")
    return data

# Power Measurements
pwr_data = load_data(cetype="QC_PWR")
for onekey in pwr_data:
    cfgdata = pwr_data[onekey]
    fembs = cfgdata[0]  # should be wibid?
    rawdata = cfgdata[1]

    for key in cfgdata[3]:
        fekey = re.sub('(FE[0-9])_.*$', '\\1', key)
        pwr_meas = kPower(measurement({key: cfgdata[3][key]}, fembid=fekey, test="Power", meastype=dict))
        pwr_meas.setConfiguration(onekey)
        serialize(pwr_meas, 'test.h5')

# FE Monitoring Data
monitoring_data = load_data(cetype="QC_MON")
for onekey in ['VBGR', 'MON_VBGR', 'MON_Temper']:
    arr = monitoring_data[onekey][1]
    for i in range(len(arr)):
        m = measurement(arr[i], fembid='FE'+str(i), test="Monitor", cfg=onekey, meastype=np.float64)
        serialize(m, 'test.h5')


AD_LSB = 2564/4096
dacdkeys = ["MON_DAC_SGP1", "MON_DAC_SG0_0_SG1_0", "MON_DAC_SG0_1_SG1_0", "MON_DAC_SG0_0_SG1_1", "MON_DAC_SG0_1_SG1_1" ]
for dacdkey in dacdkeys:
    dacs, dacv = zip(*monitoring_data[dacdkey])
    dacv_np = np.array(dacv)
    for fe in range(dacv_np.shape[1]):
        femb = 'FE'+str(fe)
        x = dacs
        y = dacv_np[:,fe]*AD_LSB
        if ("SGP1" in dacdkey) or ("SG0_1_SG1_1" in dacdkey):
            x = x[:-3]
            y = y[:-3]
        mon_meas = kDACLinearFit(measurement(zip(x, list(y)), fembid=femb, test="Monitor",
                                             cfg=dacdkey, meastype=zip))
        serialize(mon_meas, 'test.h5')

# FE Response Measurements
response_data = load_data(cetype="QC_CHKRES")
for onekey in response_data:
    cfgdata = response_data[onekey]
    fembs = cfgdata[0]
    rawdata = cfgdata[1]

    m_chns, m_rmss, m_peds, m_pkps, m_pkns = WaveformAna(fembs, rawdata,
                                                         cfg=onekey, test="Response")
    metrics = measurement({'ChanID':m_chns, 'RMS':m_rmss, 'Baseline':m_peds, 'PosPeak':m_pkps, 'NegPeak':m_pkns},
                          meastype=dict,
                          cfg=onekey, test="Response", fembid=m_chns.femb)
    serialize(metrics, 'test.h5')

# FE Noise Measurements
noise_data = load_data(cetype="QC_CHKRES")
for onekey in noise_data:
    cfgdata = noise_data[onekey]
    fembs = cfgdata[0]
    rawdata = cfgdata[1]

    m_chns, m_rmss, m_peds, _, _ = WaveformAna(fembs, rawdata,
                                                         cfg=onekey, test="Noise")
    metrics = measurement({'ChanID':m_chns, 'RMS':m_rmss, 'Baseline':m_peds},
                          meastype=dict,
                          cfg=onekey, test="Noise", fembid=m_chns.femb)
    serialize(metrics, 'test.h5')

# FE Calibration (ASIC-DAC)
caliasic_data = load_data(cetype="QC_CALI_ASICDAC")
for onekey in caliasic_data:
    cfgdata = caliasic_data[onekey]
    fembs = cfgdata[0]
    rawdata = cfgdata[1]

    m_chns, _, _, m_pkps, m_pkns = WaveformAna(fembs, rawdata,
                                                         cfg=onekey, test="Calib_ASICDAC")
    metrics = measurement({'ChanID':m_chns, 'PosPeak':m_pkps, 'NegPeak':m_pkns},
                          meastype=dict,
                          cfg=onekey, test="Calib_ASICDAC", fembid=m_chns.femb)
    serialize(metrics, 'test.h5')

# FE Calibration (DAT-DAC)
calidat_data = load_data(cetype="QC_CALI_DATDAC")
for onekey in calidat_data:
    cfgdata = calidat_data[onekey]
    fembs = cfgdata[0]
    rawdata = cfgdata[1]

    m_chns, _, m_peds, m_pkps, m_pkns = WaveformAna(fembs, rawdata,
                                                         cfg=onekey, test="Calib_DATDAC")
    metrics = measurement({'ChanID':m_chns, 'Baseline':m_peds, 'PosPeak':m_pkps, 'NegPeak':m_pkns},
                          meastype=dict,
                          cfg=onekey, test="Calib_DATDAC", fembid=m_chns.femb)
    serialize(metrics, 'test.h5')

# FE Calibration (Direct-Input)
calidirect_data = load_data(cetype="QC_CALI_DIRECT")
for onekey in calidirect_data:
    cfgdata = calidirect_data[onekey]
    fembs = cfgdata[0]
    rawdata = cfgdata[1]

    m_chns, _, m_peds, m_pkps, m_pkns = WaveformAna(fembs, rawdata,
                                                         cfg=onekey, test="Calib_DIRECT")
    metrics = measurement({'ChanID':m_chns, 'Baseline':m_peds, 'PosPeak':m_pkps, 'NegPeak':m_pkns},
                          meastype=dict,
                          cfg=onekey, test="Calib_DIRECT", fembid=m_chns.femb)
    serialize(metrics, 'test.h5')

#  FE Delay Run
delay_data = load_data(cetype="QC_DLY_RUN")
wfs_delay = []
m_chns = None
#  check interleaving code for what phases represent -- this seems to work for now
#  for onekey in delay_data:
for i in list(range(23, 32))+list(range(0, 23)):
    onekey = "Phase00%.2d"%i + "_freq1000"
    cfgdata = delay_data[onekey]
    fembs = cfgdata[0]
    rawdata = cfgdata[1]

    m_chns, _, _, _, _, wfs = WaveformAna(fembs, rawdata,
                                     cfg=onekey, test="DelayRun", getwfs=True)
    wfs_delay.append(wfs)

wfs_delay = np.transpose(np.array(wfs_delay), (1, 2, 0))
wfs_delay = np.flip(wfs_delay, axis=2)
wfs_delay = wfs_delay.reshape(wfs_delay.shape[0], -1)
m_wfs = waveform(wfs_delay, cfg="Waveform", fembid=m_chns.femb, test="DelayRun")
m_chns.setConfiguration("ChanID")
serialize(m_chns, 'test.h5')
serialize(m_wfs, 'test.h5')

# FE Cali-Cap Measurement
cap_data = load_data(cetype="QC_Cap_Meas")
pps4s = [None]*16
vals = [None]*16
for onekey in cap_data:
    cfgdata = cap_data[onekey]
    fembs = cfgdata[0]
    rawdata = cfgdata[1]
    fembchn = cfgdata[2]
    val =     cfgdata[3]
    period =  cfgdata[4]
    width =   cfgdata[5]
    fe_info = cfgdata[6]
    cfg_info = cfgdata[7]

    chnc, pps = CapacitanceAna(fembs, rawdata, onekey)
    assert chnc == fembchn, "Didn't match the right LArASIC channel!"

    if not pps4s[chnc]:
        pps4s[chnc] = []
        vals[chnc] = []
    pps4s[chnc].append(pps)
    vals[chnc].append(val)

pps4s = np.array(pps4s)
vals = np.array(vals)

ratios = (pps4s[:,0,:] - pps4s[:,1,:])/(pps4s[:,2,:] - pps4s[:,3,:])
del1 = vals[0][3] - vals[0][2]
del2 = vals[0][1] - vals[0][0]
caps = ratios.flatten('F')*del1/del2/0.185
m_caps = measurement(caps, meastype=np.ndarray,
                     fembid='CE0', cfg='CapacitancePerPC', test="Calib_Capacitance")
serialize(m_caps, 'test.h5')
