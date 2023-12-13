import os
import sys
import pickle
import numpy as np
import re

sys.path.append('../decode')
from spymemory_decode import wib_dec

sys.path.append('../core')
from measure import *
from ceio import *

sys.path.append('../ana')
from measurements import *

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
