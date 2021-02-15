import numpy as np
import datetime
import logging
import sys
import os
from sarhspredictor.lib.reference_oswk import reference_oswK_954m_30pts,reference_oswK_954m_60pts,reference_oswK_1145m_60pts
def patch_oswK(k_sar,ipfvesion=None,datedtsar=None):
    """
    :args:
        k_sar (nd.array): oswK content
        ipfvesion (str): '002.53' for instance
        datedtsar (datetime): start date of the product considered
    #first value seen in oswK:
    20200101 -> 60 pts 0.005235988 IPF 3.10
    20190101 -> 60 pts 0.005235988 IPF 2.91
    20180101 -> 60 pts 0.005235988 IPF 2.84
    20170101 -> 60 pts 0.005235988 IPF 2.72
    20151125 -> 60 pts 0.005235988 IPF 2.60
    20150530T195229 -> passage 2.53 vers 2.60 -> oswK extended from 954m up to 1145m wavelength
    20151124 -> 60 pts 0.006283185 IPF 2.53
    20150530T195229 -> passage 2.43 vers 2.53 -> resolution 72x60
    20150530 -> 30 pts 0.006283185 IPF 2.43
    20150203 -> 30 pts 0.006283185 IPF 2.36
    to patch for oswK vectors from WV S-1 osw products that could contains NaN or masked values
    :return:
    """
    if isinstance(k_sar,np.ma.core.MaskedArray):
        test_oswk_KO = (k_sar.mask == True).any()
    else:
        #isinstance(k_sar,np.ndarray):
        logging.debug('nd.array case : %s',np.isfinite(k_sar))
        test_oswk_KO = (np.isfinite(k_sar)==False).any()# or (k_sar.mask==True).any()

    if test_oswk_KO:
        logging.debug('oswK is corrupted : %s',k_sar)
        if len(k_sar) == 30 :  # starting from 3 july 2015 (+1 month ok: june 2015) in we have 60 elements in oswK but still some oswK can contains erroneous masked values
            # k_sar = reference_oswK_2015
            k_sar = reference_oswK_954m_30pts
            logging.info('ref k 30 elements (instead of 60) %s',k_sar.shape)
        else :
            if ipfvesion is not None:
                if ipfvesion in ['002.53']:
                    k_sar = reference_oswK_954m_60pts
                else:
                    k_sar = reference_oswK_1145m_60pts
            else:
                if datedtsar < datetime.datetime(2015,11,24,19,52,29) :  # date of IPF 2.53 -> 2.60
                    # if sar['fdatedt']<datetime.datetime(2018,1,1) and sar['fdatedt']>=datetime.datetime(2017,1,1):
                    # k_sar = reference_oswK_2017
                    k_sar = reference_oswK_954m_60pts
                else :
                    k_sar = reference_oswK_1145m_60pts  # latest version of oswK on 60 points
        logging.debug('corrected k = %s',k_sar)
    else:
        logging.debug('nothing to do oswK is clean')
    return k_sar


if __name__ =='__main__':
    logging.basicConfig(level=logging.DEBUG)
    import netCDF4
    import os
    file_ok = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2019/339/S1A_WV_OCN__2SSV_20191205T083252_20191205T084128_030211_037400_D82C.SAFE/measurement/s1a-wv1-ocn-vv-20191205t083519-20191205t083522-030211-037400-011.nc'
    file_ko = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2015/017/S1A_WV_OCN__2SSV_20150117T144433_20150117T144436_004212_0051E0_0D74.SAFE/measurement/s1a-wv2-ocn-vv-20150117t141953-20150117t144436-004212-0051E0-074.nc'
    file_ko2  = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2017/033/S1A_WV_OCN__2SSV_20170202T072343_20170202T073839_015101_018B06_972C.SAFE/measurement/s1a-wv1-ocn-vv-20170202t073132-20170202t073135-015101-018b06-033.nc'
    for ff in [file_ok,file_ko,file_ko2]:
        nc = netCDF4.Dataset(ff)
        sardatedt = datetime.datetime.strptime(os.path.basename(ff).split('-')[4],'%Y%m%dt%H%M%S')
        input_ksar = nc.variables['oswK'][:]
        k_sar = patch_oswK(input_ksar,sardatedt)
        # logging.info('input k %s output %s',input_ksar,k_sar)
        nc.close()
