# Reads NetCDF4 file, preprocesses data, and writes hdf5 file.
# This is much simpler than aggregating multiple files, then
# performing preprocessing.
# Author: Peter Sadowski, Dec 2020
import os, sys, h5py
import numpy as np
import glob
import time
from dateutil import rrule
import traceback
import xarray
import logging
import datetime
import netCDF4
from sarhspredictor.lib.sarhs import preprocess

# Source and destination filenames.
#file_src  = "/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/S1B_201905_test01S/S1B_201905_test01S.nc"  # Example file containing single observation.
#file_dest = "/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/S1B_201905_test01S/S1B_201905_test01S_processed.h5"
#file_src = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/quach2020/validation/input_output/final2/S1B_20190501_ifr_tmp_input_output_quach2020_pythonv2.nc'
#file_src = '/home1/datawork/agrouaze/data/sentinel1/cwave/training_dataset_quach2020_python_v2/S1A_ALT_coloc201501S.nc'

# preprocessing method for the new coloc files WV vs ALTI (based on cwaveV4) but with SLC x spectra
def prep(ds):
    #ds = ds.drop('py_S') #bug on the run for 2015 16h the 2 June 2021
    ds = ds.drop('dk')
    ds = ds.drop('k')
    return ds

# reference_oswK_1145m_60pts
kref = np.array([0.005235988, 0.00557381, 0.005933429, 0.00631625, 0.00672377,
    0.007157583, 0.007619386, 0.008110984, 0.008634299, 0.009191379,
    0.0097844, 0.01041568, 0.0110877, 0.01180307, 0.01256459, 0.01337525,
    0.01423822, 0.01515686, 0.01613477, 0.01717577, 0.01828394, 0.01946361,
    0.02071939, 0.02205619, 0.02347924, 0.02499411, 0.02660671, 0.02832336,
    0.03015076, 0.03209607, 0.03416689, 0.03637131, 0.03871796, 0.04121602,
    0.04387525, 0.04670605, 0.0497195, 0.05292737, 0.05634221, 0.05997737,
    0.06384707, 0.06796645, 0.0723516, 0.07701967, 0.08198893, 0.08727881,
    0.09290998, 0.09890447, 0.1052857, 0.1120787, 0.1193099, 0.1270077,
    0.1352022, 0.1439253, 0.1532113, 0.1630964, 0.1736193, 0.1848211,
    0.1967456, 0.2094395])
def normalize_training_ds(sta,sto,in_dd,out_dd,redo=False):
    """

    :param sta: datetime
    :param sto: datetime
    :param in_dd: str where the colocs SAR and Alti are stored
    :param out_dd: str where the same dataset but normalized will be stored
    :return:
    """
    if os.path.exists(out_dd) is False:
        os.makedirs(out_dd,0o0775)
        logging.info('makedir %s',out_dd)
    for sat in ['S1A','S1B']:
        #lst_training_files = glob.glob(os.path.join('/home1/datawork/agrouaze/data/sentinel1/cwave/training_dataset_quach2020_python_v2/',sat+'*.nc'))
        if sat=='S1A':
            satellite = 1 # 1=S1A, 0=S1B
        else:
            satellite = 0
        #print('nb input files to train for %s : %s'%(sat,len(lst_training_files)))
        #for ffii,file_src in enumerate(lst_training_files):
        for mm in rrule.rrule(rrule.MONTHLY,dtstart=sta,until=sto):
            #file_dest = os.path.join(out_dd,os.path.basename(file_src).replace('.nc','_processed.nc'))
            pat = os.path.join(in_dd,mm.strftime('%Y'),'*','training_'+sat.lower()+'*vv-%s*.nc'%mm.strftime('%Y%m'))
            file_dest = os.path.join(out_dd,sat+'_training_exp1_Hs_NN_regression_%s.h5'%mm.strftime('%Y%m'))
            if os.path.exists(file_dest) and redo is True:
                os.remove(file_dest)
                logging.info('remove existing output file : %s',file_dest)
            if os.path.exists(file_dest) and redo is False:
                logging.info('%s already exists -> skip and continue the month loop',file_dest)
                continue
            lst_files_measu_to_read = glob.glob(pat)#[0:10] # for dev !!!!!!!!!!!
            logging.info('nb files for month %s : %s',mm,len(lst_files_measu_to_read))
            if len(lst_files_measu_to_read)>0:
                print('file_dest',file_dest,'month',mm)
                if os.path.exists(os.path.dirname(file_dest)) is False:
                    os.makedirs(os.path.dirname(file_dest))
                    print('outputdir mkdir')
                # These variables are expected in the source file.
                keys = ['timeSAR', 'lonSAR',  'latSAR', 'incidenceAngle', 'crossSpectraRePol', 'crossSpectraImPol',
                        'py_S','sigma0','normalizedVariance'] # Needed for predictions.
                t0 = time.time()
                # try:
                #     h5py.File(file_dest, 'r').close() #try to close the file if it is opened before
                # except:
                #     print('traceback',traceback.format_exc())
                #     pass
                src = xarray.open_mfdataset(lst_files_measu_to_read,combine='by_coords',concat_dim='time',
                                            preprocess=prep,cache=True,decode_times=True) #,preprocess=None
                src = src.assign_coords({'k':kref})
                logging.info('src: %s',src)
                #with Dataset(file_src) as fs, h5py.File(file_dest, 'w') as fd:
                # Check input file.
                #src = fs.variables
                with h5py.File(file_dest, 'w') as fd:
                    for k in keys:
                        if k not in src:
                            raise IOError(f'Variable {k} not found in input file.')
                    num_examples = src[keys[0]].shape[0]
                    print(f'Found {num_examples} events.')

                    # Get 22 CWAVE features. Concatenate 20 parameters with sigma0 and normVar.
                    #src['S'].set_auto_scale(False) # Some of the NetCDF4 files had some weird scaling.
                    S = np.array(src['py_S'].values) #* float(src['py_S'].scale_factor))
                    cwave = np.hstack([S, src['sigma0'].values.reshape(-1,1), src['normalizedVariance'].values.reshape(-1,1)])
                    #cwave = src['cwave'][:]
                    cwave = preprocess.conv_cwave(cwave) # Remove extrema, then standardize with hardcoded mean, vars.
                    fd.create_dataset('cwave', data=cwave)

                    # Observation meta data.
                    latSAR, lonSAR = src['latSAR'].values, src['lonSAR'].values
                    latSARcossin = preprocess.conv_position(latSAR) # Computes cos and sin used by NN.
                    lonSARcossin = preprocess.conv_position(lonSAR)
                    #latlonSARcossin = src['latlonSARcossin'].values
                    fd.create_dataset('latlonSAR', data=np.column_stack([latSAR, lonSAR]))
                    fd.create_dataset('latlonSARcossin', data=np.hstack([latSARcossin, lonSARcossin]))
                    #fd.create_dataset('latlonSARcossin', data=latlonSARcossin)

                    timeSAR = src['timeSAR'].values
                    unit = 'hours since 2010-01-01T00:00:00Z UTC' #asked by P.Sadowsky routine
                    logging.info("timeSAr units: %s",unit)
                    time_num = []
                    for tti in timeSAR:
                        ts = (tti - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
                        ts2 =  datetime.datetime.utcfromtimestamp(ts)
                        tnum = netCDF4.date2num(ts2,unit)
                        time_num.append(tnum)
                    timeSAR = np.array(time_num)

                    # idem for timeALT
                    timeALT = src['timeALT'].values  # added by agrouaze
                    time_num = []
                    for tti in timeALT :
                        ts = (tti - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
                        ts2 = datetime.datetime.utcfromtimestamp(ts)
                        tnum = netCDF4.date2num(ts2,unit)
                        time_num.append(tnum)
                    timeALT = np.array(time_num)
                    #timeSAR = np.datetime64(timeSAR.astype('float'),'s')
                    logging.info('timeSAR %s %s',timeSAR[0],type(timeSAR[0]))
                    todSAR = preprocess.conv_time(timeSAR)
                    #todSAR = src['todSAR'].values
                    fd.create_dataset('timeSAR', data=timeSAR, shape=(timeSAR.shape[0], 1))
                    fd.create_dataset('todSAR', data=todSAR, shape=(todSAR.shape[0], 1))

                    incidence = preprocess.conv_incidence(src['incidenceAngle'].values) # Separates into 2 var.
                    fd.create_dataset('incidence', data=incidence)

                    satellite_indicator = np.ones((src['timeSAR'].shape[0], 1), dtype=float) * satellite
                    fd.create_dataset('satellite', data=satellite_indicator, shape=(satellite_indicator.shape[0], 1))

                    # Spectral data.
                    logging.debug('x spec Re: %s',src['crossSpectraRePol'].values.shape)
                    tmpre = src['crossSpectraRePol'].values.squeeze()
                    tmpre = np.swapaxes(tmpre,1,2)
                    tmpim = src['crossSpectraImPol'].values.squeeze()
                    tmpim = np.swapaxes(tmpim,1,2)
                    logging.debug('tmpre : %s',tmpre.shape)
                    re = preprocess.conv_real(tmpre,exp_id=1)
                    im = preprocess.conv_imaginary(tmpim,exp_id=1)
                    x = np.stack((re, im), axis=3)
                    logging.debug('spectrum : %s',x.shape)
                    fd.create_dataset('spectrum', data=x) # spectrum doesnt seems to be used at the next preprocessing step ( aggregate_monthly_training_files.aggregate() )

                    # Altimeter features.
                    hsALT = src['hsALT'].values
                    fd.create_dataset('hsALT', data=hsALT, shape=(hsALT.shape[0], 1))
                    dx = preprocess.conv_dx(src['dx'].values)
                    dt = preprocess.conv_dt(src['dt'].values)
                    fd.create_dataset('dxdt', data=np.column_stack([dx, dt]))


                    fd.create_dataset('timeALT',data=timeALT, shape=(todSAR.shape[0], 1))

                    lonALT = src['lonALT'].values #added by agrouaze
                    fd.create_dataset('lonALT', data=lonALT)

                    latALT = src['latALT'].values #added by agrouaze
                    fd.create_dataset('latALT', data=latALT)

                    fd.create_dataset('hsSM', data=src['hsSM'].values) #added by agrouaze
                    fd.create_dataset('nk', data=src['nk'].values) #added by agrouaze
                    fd.create_dataset('dx', data=src['dx'].values) #added by agrouaze
                    fd.create_dataset('dt', data=src['dt'].values) #added by agrouaze
                    fd.create_dataset('sigma0', data=src['sigma0'].values) #added by agrouaze
                    fd.create_dataset('normalizedVariance', data=src['normalizedVariance'].values) #added by agrouaze
                    fd.create_dataset('incidenceAngle', data=src['incidenceAngle'].values) #added by agrouaze
                    fd.create_dataset('lonSAR', data=src['lonSAR'].values) #added by agrouaze
                    fd.create_dataset('latSAR', data=src['latSAR'].values) #added by agrouaze
                    fd.create_dataset('cspcRe', data=src['crossSpectraRePol'].values) #added by agrouaze
                    fd.create_dataset('cspcIm', data=src['crossSpectraImPol'].values) #added by agrouaze
                    fd.create_dataset('py_S', data=S) #added by agrouaze
                    #fd.close()
                print('elapsed time to build %s: %1.3f seconds'%(file_dest,time.time()-t0))
                logging.info('fiel written %s',os.path.exists(file_dest))
