import sys
import numpy as np
import statsmodels.api as sm
import re

sys.path.append('../core')
from measure import *

sys.path.append('../decode')
from spymemory_decode import wib_dec

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
