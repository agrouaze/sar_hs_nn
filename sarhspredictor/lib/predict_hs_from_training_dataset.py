"""
Feb 2021
A. Grouazel
script to validate that I can get same Hs and Hs STD from OCN file sand training dataset (using the same model)
"""
import os
import logging
import numpy as np
import xarray
import netCDF4
import pdb
import datetime
from sarhspredictor.lib.load_quach_2020_keras_model import load_quach2020_model_v2
from sarhspredictor.lib.sarhs import preprocess
from sarhspredictor.lib.sarhs.preprocess import conv_time
from sarhspredictor.lib.predict_with_quach2020_on_OCN_using_keras import do_my_prediction,define_features,define_input_test_dataset
def from_np64_to_dt(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)

def prepare_training_dataset_core(ds_train_raw,validation_dataset=False):
    """
    this method I used for building training dataset and also to do the validation dataset
    :param ds_train_raw:
    :return:
    """
    # except: #for py2.7 version
    #    ocn_wv_ds = xarray.open_mfdataset(pattern_path,concat_dim='time',preprocess=preproc_ocn_wv)
    logging.info('Nb pts in dataset: %s',ds_train_raw['timeSAR'].size)
    varstoadd = ['S','cwave','dxdt','latlonSARcossin','todSAR',
                 'incidence','satellite','oswQualityCrossSpectraRe','oswQualityCrossSpectraIm']
    # additional_vars_for_validation = ['oswLon','oswLat','oswLandFlag','oswIncidenceAngle','oswWindSpeed','platformName',
    #                                  'nrcs','nv','heading','oswK','oswNrcs']
    # varstoadd += additional_vars_for_validation
    if validation_dataset:
        varstoadd.append('py_cspcImX')
        varstoadd.append('py_cspcReX')
    if 'hsSM' in ds_train_raw:
        varstoadd += ['hsSM']
    S = ds_train_raw['py_S'].values
    s0 = ds_train_raw['sigma0']
    nv = ds_train_raw['normalizedVariance'].values
    ds_training_normalized = xarray.Dataset()
    timeSAR_vals = ds_train_raw['timeSAR'].values
    ths1 = ds_train_raw['th'].values
    ks1 = ds_train_raw['k'].values
    if 'fileNameFull' in ds_train_raw:
        fpaths = ds_train_raw['fileNameFull'].values
        #varstoadd.append('fileNameFull')
    else:
        fpaths = ds_train_raw['fileNameL2'].values # 2019 dataset is a bt different
        #varstoadd.append('fileNameL2')
    sattelites = np.array([os.path.basename(hhy)[0 :3] for hhy in fpaths])
    satellites_int = np.array([threelettersat[2] == 'a' for threelettersat in sattelites]).astype(int)
    cspcRe = ds_train_raw['cspcRe'].values
    cspcIm = ds_train_raw['cspcIm'].values
    for vv in varstoadd :
        logging.info('start format variable :%s',vv)
        if vv in ['cwave'] :
            dimszi = ['time','cwavedim']
            coordi = {'time' : timeSAR_vals,'cwavedim' : np.arange(22)}
            logging.debug('S %s s0: %s nv: %s',S.shape,s0.shape,nv.shape)
            cwave = np.vstack([S.T,s0,nv]).T  # found L77 in preprocess.py
            logging.debug('cwave vals: %s',cwave.shape)
            cwave = preprocess.conv_cwave(cwave)
            ds_training_normalized[vv] = xarray.DataArray(data=cwave,dims=dimszi,coords=coordi)
        elif vv in ['fileNameFull','fileNameL2']:
            dimszi = ['time','pathnchar']
            coordi = {'time' : timeSAR_vals,'pathnchar' : len(fpaths[0])}
            ds_training_normalized[vv] = xarray.DataArray(data=fpaths,dims=dimszi,coords=coordi)
        elif vv == 'S' :  # to ease the comparison with Justin files
            dimszi = ['time','Sdim']
            coordi = {'time' : timeSAR_vals,'Sdim' : np.arange(20)}
            ds_training_normalized[vv] = xarray.DataArray(data=S,dims=dimszi,coords=coordi)
        elif vv in ['dxdt'] :  # dx and dt and delta from coloc with alti see /home/cercache/users/jstopa/sar/empHs/cwaveV5, I can put zeros here at this stage
            #dxdt = np.column_stack([ds_train_raw['dx'].values,ds_train_raw['dt'].values])
            dxdt = np.column_stack([np.zeros(s0.shape),np.ones(s0.shape)])
            dimszi = ['time','dxdtdim']
            coordi = {'time' : timeSAR_vals,'dxdtdim' : np.arange(2)}
            ds_training_normalized[vv] = xarray.DataArray(data=dxdt,dims=dimszi,coords=coordi)
        elif vv in ['latlonSARcossin'] :
            latSARcossin = preprocess.conv_position(ds_train_raw['latSAR'].values)  # Gets cos and sin
            lonSARcossin = preprocess.conv_position(ds_train_raw['lonSAR'].values)
            latlonSARcossin = np.hstack([latSARcossin,lonSARcossin])
            dimszi = ['time','latlondim']
            coordi = {'time' : timeSAR_vals,'latlondim' : np.arange(4)}
            ds_training_normalized[vv] = xarray.DataArray(data=latlonSARcossin,dims=dimszi,coords=coordi)
        elif vv in ['todSAR'] :
            dimszi = ['time']
            new_dates_dt = np.array([from_np64_to_dt(dt64) for dt64 in timeSAR_vals])
            unit = "hours since 2010-01-01T00:00:00Z UTC"  # see https://github.com/grouny/sar_hs_nn/blob/c05322e6635c6d77409e36537d7c3b58788e7322/sarhspredictor/lib/sarhs/preprocess.py#L11
            new_dates_num = np.array([netCDF4.date2num(dfg,unit) for dfg in new_dates_dt])
            coordi = {'time' : timeSAR_vals}
            todSAR = conv_time(new_dates_num)
            ds_training_normalized[vv] = xarray.DataArray(data=todSAR,dims=dimszi,coords=coordi)
        elif vv in ['oswK'] :
            dimszi = ['time','oswWavenumberBinSize']
            coordi = {'time' : timeSAR_vals,'oswWavenumberBinSize' : np.arange(len(ks1))}
            ds_training_normalized[vv] = xarray.DataArray(data=ks1,dims=dimszi,coords=coordi)
        elif vv in ['incidence',] :
            dimszi = ['time','incdim']
            coordi = {'time' : timeSAR_vals,'incdim' : np.arange(2)}
            incidence = preprocess.conv_incidence(ds_train_raw['incidenceAngle'].values.squeeze())
            ds_training_normalized[vv] = xarray.DataArray(data=incidence,dims=dimszi,coords=coordi)
        elif vv in ['incidence_angle'] :
            dimszi = ['time']
            olddims = [x for x in ds_train_raw['incidenceAngle'].dims if x not in ['oswAzSize','oswRaSize']]
            coordi = {}
            for didi in olddims :
                coordi[didi] = ds_train_raw['incidenceAngle'].coords[didi].values
            coordi['time'] = timeSAR_vals
            incidence = np.array([ds_train_raw['incidenceAngle'].values.squeeze()])
            ds_training_normalized[vv] = xarray.DataArray(data=incidence,dims=dimszi,coords=coordi)
        elif vv in ['satellite'] :
            dimszi = ['time']
            coordi = {'time' : timeSAR_vals}
            # satellite_int = np.array([satellite[2] == 'a']).astype(int)
            ds_training_normalized[vv] = xarray.DataArray(data=satellites_int,dims=dimszi,coords=coordi)
        elif vv in ['platformName'] :
            dimszi = ['time']
            coordi = {'time' : timeSAR_vals}
            satellite_int = sattelites
            ds_training_normalized[vv] = xarray.DataArray(data=satellite_int,dims=dimszi,coords=coordi)
        elif vv in ['nrcs'] :
            dimszi = ['time']
            coordi = {'time' : timeSAR_vals}
            ds_training_normalized[vv] = xarray.DataArray(data=s0,dims=dimszi,coords=coordi)
        elif vv in ['heading'] :
            dimszi = ['time']
            coordi = {'time' : timeSAR_vals}
            ds_training_normalized[vv] = xarray.DataArray(data=ds_train_raw['trackAngle'].values,dims=dimszi,
                                                          coords=coordi)
        elif vv in ['nv'] :
            dimszi = ['time']
            coordi = {'time' : timeSAR_vals}
            ds_training_normalized[vv] = xarray.DataArray(data=nv,dims=dimszi,coords=coordi)
        elif vv in ['oswQualityCrossSpectraRe','oswQualityCrossSpectraIm'] :
            if vv == 'oswQualityCrossSpectraRe' :
                datatmp = cspcRe
            elif vv == 'oswQualityCrossSpectraIm' :
                datatmp = cspcIm
            else :
                raise Exception()
            # datatmp = ds[vv].values.squeeze()
            # olddims = [x for x in ds[vv].dims if x not in ['oswAzSize','oswRaSize']]
            coordi = {}
            # for didi in olddims:
            #    coordi[didi] = ds[vv].coords[didi].values
            coordi['time'] = timeSAR_vals
            coordi['oswAngularBinSize'] = np.arange(len(ths1))
            coordi['oswWavenumberBinSize'] = np.arange(len(ks1))
            dimsadd = ['time','oswAngularBinSize','oswWavenumberBinSize']
            # if datatmp.shape == (72,60) :  # case only one spectra
            #    datatmp = datatmp.reshape((1,72,60))

            ds_training_normalized[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        elif vv in ['py_cspcImX','py_cspcReX'] :
            datatmp = ds_train_raw[vv].values
            coordi = ds_train_raw[vv].coords
            coordi['time'] = timeSAR_vals
            dimsadd = ds_train_raw[vv].dims
            ds_training_normalized[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        else :
            datatmp = ds_train_raw[vv].values.squeeze()
            olddims = [x for x in ds_train_raw[vv].dims if x not in ['oswAzSize','oswRaSize']]
            coordi = {}
            for didi in olddims :
                coordi[didi] = ds_train_raw[vv].coords[didi].values
            coordi['time'] = timeSAR_vals
            dimsadd = ['time']
            logging.info('data: %s',datatmp.shape)
            ds_training_normalized[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        # logging.debug('field xarray : %s %s',vv,newds[vv])
    logging.debug('newds: %s',ds_training_normalized)
    logging.info('SAR data ready to be used')
    # cspcRe = ds_train_raw['oswQualityCrossSpectraRe'].values
    # cspcIm = ds_train_raw['oswQualityCrossSpectraIm'].values
    re = preprocess.conv_real(cspcRe)
    im = preprocess.conv_imaginary(cspcIm)
    logging.info('re : %s',re.shape)
    logging.info('im : %s',im.shape)
    spectrum = np.stack((re,im),axis=3)
    logging.info('spectrum shape : %s',spectrum.shape)
    return spectrum,ds_training_normalized

def prepare_training_dataset(pattern_path):
    """
    read training dataset and normalize the variable needed by the model
    :param pattern_path: could also be a list of path
    :return:
    """
    #ff = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2020/129/*.SAFE/measurement/s1*nc'
    logging.info('start reading training dataset')
    #try:
    #ocn_wv_ds = xarray.open_mfdataset(pattern_path,combine='by_coords',concat_dim='time',preprocess=preproc_ocn_wv)
    ds_train_raw = xarray.open_dataset(pattern_path)
    spectrum_x,ds_training_normalized_x = prepare_training_dataset_core(ds_train_raw)
    return spectrum_x,ds_training_normalized_x

def compute_prediction_from_training_inputs(ds_training,spectrum,model):
    features = define_features(ds_training)
    test = define_input_test_dataset(features,spectrum)
    yhat = do_my_prediction(model,test)
    ds_training['swh'] = xarray.DataArray(data=yhat[:,0],dims=['time'],coords={'time':ds_training['time'].values})
    ds_training['swh_uncertainty'] = xarray.DataArray(data=yhat[:,1],dims=['time'],coords={'time':ds_training['time'].values})
    return ds_training

if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='predict from training for validation')
    parser.add_argument('--verbose',action='store_true',default=False)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    pattern_path = '/home1/datawork/agrouaze/data/sentinel1/cwave/training_dataset_quach2020_python_v2/S1A_ALT_coloc201501S.nc'
    pattern_path = '/home1/datawork/agrouaze/data/sentinel1/cwave/training_dataset_quach2020_python_v2/S1A_ALT_coloc201506S.nc'
    spectrum,ds_train_norm = prepare_training_dataset(pattern_path)


    heteroskedastic_2017 = load_quach2020_model_v2()
    ds_train_with_swh = compute_prediction_from_training_inputs(ds_train_norm,spectrum,heteroskedastic_2017)
    logging.info('ds_training swh: %sm std:%s',ds_train_with_swh['swh'].values,ds_train_with_swh['swh_uncertainty'].values)