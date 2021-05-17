#!/usr/bin/env python
"""

Dec 2020:
"""
import os, datetime
import socket
from IPython import get_ipython
HOSTNAME = socket.gethostname()
INTERACTIVE = get_ipython() is not None
if INTERACTIVE:
    get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')
SHERPA_TRIAL_ID = os.environ.get('SHERPA_TRIAL_ID', '0')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Needed to avoid cudnn bug.
import logging
import numpy as np
import xarray
from sarhspredictor.lib.sarhs import preprocess
from sarhspredictor.lib.preproc_ocn_wv import preproc_ocn_wv


def prepare_ocn_wv_data(pattern_path):
    """
    :param pattern_path: could also be a list of path
    :return:
    """
    #ff = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2020/129/*.SAFE/measurement/s1*nc'
    logging.info('start reading S1 WV OCN data')
    #try:
    ocn_wv_ds = xarray.open_mfdataset(pattern_path,combine='by_coords',concat_dim='time',preprocess=preproc_ocn_wv)
    #except: #for py2.7 version
    #    ocn_wv_ds = xarray.open_mfdataset(pattern_path,concat_dim='time',preprocess=preproc_ocn_wv)
    logging.info('Nb pts in dataset: %s',ocn_wv_ds['todSAR'].size)
    logging.info('SAR data ready to be used')
    cspcRe = ocn_wv_ds['oswQualityCrossSpectraRe'].values
    cspcIm = ocn_wv_ds['oswQualityCrossSpectraIm'].values

    re = preprocess.conv_real(cspcRe)
    im = preprocess.conv_imaginary(cspcIm)
    logging.info('re : %s',re.shape)
    logging.info('im : %s',im.shape)
    spectrum = np.stack((re, im), axis=3)
    logging.info('spectrum shape : %s',spectrum.shape)
    return spectrum,ocn_wv_ds

def define_features(ds):
    """
    :param ds: xarrayDataArray of OCN WV data
    :return:
    """
    features = None
    for jj in ['cwave','dxdt','latlonSARcossin','todSAR','incidence','satellite'] :
        addon = ds[jj].values
        if len(addon.shape) == 1 :
            addon = addon.reshape((addon.size,1))
        if features is None :
            features = addon
            logging.debug('add %s features: %s',jj,features.shape)
        else :
            features = np.hstack([features,addon])
            logging.debug('add %s features: %s',jj,features.shape)
    #  = np.dstack([ds['cwave'].values,ds['dxdt'].values,ds['latlonSARcossin'].values,ds['todSAR'].values,ds['incidence'].values,ds['satellite'].values])
    logging.debug('features ready')
    return features

def define_input_test_dataset(features,spectrum):
    """

    :param features: 2D np matrix
    :param spectrum: 4D np matrix
    :return:
    """
    outputs = np.zeros(features.shape[0])
    inputs = [spectrum,features]
    test = (inputs,outputs)
    logging.debug('test dataset ready')
    return test

def predict ( model,dataset ) :
    """

    :param model:
    :param dataset:
    :return:
    """
    ys,yhats = [],[]
    for batch in dataset :
        inputs,y = batch
        print('inputs',type(inputs))
        print('y',y)
        yhat = model.predict_on_batch(inputs)
        if y is not None :
            y = y.reshape(-1,2)
        else :
            y = np.zeros((yhat.shape[0],1))
        ys.append(y)
        yhats.append(yhat)
    yhat = np.vstack(yhats)
    y = np.vstack(ys)
    return y,yhat

# my version without batch
def predict_agrouaze ( model,dataset ) :
    ys,yhats = [],[]
    inputs,yhats_vide = dataset
    yhat = model.predict_on_batch(inputs)
    logging.debug('yhat %s %s',yhat.shape,yhat)
    # if y is not None :
    #     y = y.reshape(-1,2)
    #     pass
    # else :
    #     y = np.zeros((yhat.shape[0],1))
    # ys.append(y)
    yhats.append(yhat)
    yhat = np.vstack(yhats)
    #y = np.vstack(ys)
    return yhat

def do_my_prediction(model,test):
    """

    :param model:
    :param test:
    :return:
    """
    from tqdm import tqdm
    logging.debug('start Hs prediction')
    # _, yhat = predict(model,test)
    yhat = predict_agrouaze(model,test)
    logging.debug('prediction finished')
    return yhat


def main_level_0(s1_ocn_wv_ds,spectrum,model):
    """

    :param s1_ocn_wv_ds: xarray data array read by prepare_ocn_wv_data()
    :return:
    """
    features = define_features(s1_ocn_wv_ds)
    test = define_input_test_dataset(features,spectrum)
    yhat = do_my_prediction(model,test)
    s1_ocn_wv_ds['HsQuach'] = xarray.DataArray(data=yhat[:,0],dims=['time'])
    s1_ocn_wv_ds['HsQuach_uncertainty'] = xarray.DataArray(data=yhat[:,1],dims=['time'])
    return s1_ocn_wv_ds

def main_level_1(pattern_path,model):
    """
    :param pattern_path: (str) or list of str path
    :return:
    """

    spectrum,s1_ocn_wv_ds = prepare_ocn_wv_data(pattern_path)
    features = define_features(s1_ocn_wv_ds)
    test = define_input_test_dataset(features,spectrum)
    yhat = do_my_prediction(model,test)
    #s1_ocn_wv_ds['HsQuach'] = xarray.DataArray(data=yhat[:,0],dims=['time'])
    #s1_ocn_wv_ds['HsQuach_uncertainty'] = xarray.DataArray(data=yhat[:,1],dims=['time'])
    s1_ocn_wv_ds['swh'] = xarray.DataArray(data=yhat[:,0],dims=['time'])
    s1_ocn_wv_ds['swh_uncertainty'] = xarray.DataArray(data=yhat[:,1],dims=['time'])
    return s1_ocn_wv_ds
