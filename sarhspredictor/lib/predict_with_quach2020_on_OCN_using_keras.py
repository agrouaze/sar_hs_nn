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
#import sherpa
import numpy as np
#import pandas as pd
#import h5py
#from pathlib import Path
#from shutil import copyfile
#import tensorflow as tf
#from tensorflow.keras.callbacks import *
#from tensorflow.keras.layers import concatenate
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.utils import plot_model
#import tensorflow.keras as keras
import sys
sys.path.append('/home1/datahome/agrouaze/git/SAR-Wave-Height/')
#import sarhs.generator
#import importlib

import xarray
#import datetime
from sarhs import preprocess

#from load_quach_2020_keras_model import load_quach2020_model
#from load_quach_2020_keras_model import POSSIBLES_MODELS
from preproc_ocn_wv import preproc_ocn_wv

# try:
#     print('first try to  load_quach2020_model')
#     MODEL = load_quach2020_model()
# except:
#     print('second try to  load_quach2020_model (except)')
#     MODEL = load_quach2020_model()



def prepare_ocn_wv_data(pattern_path):
    """
    :param pattern_path: could also be a list of path
    :return:
    """
    #ff = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2020/129/*.SAFE/measurement/s1*nc'
    logging.info('start reading S1 WV OCN data')
    ocn_wv_ds = xarray.open_mfdataset(pattern_path,combine='by_coords',concat_dim='time',preprocess=preproc_ocn_wv)
    logging.info('Nb pts in dataset: %s',ocn_wv_ds['todSAR'].size)
    logging.info('SAR data ready to be used')
    cspcRe = ocn_wv_ds['oswQualityCrossSpectraRe'].values.squeeze()
    cspcIm = ocn_wv_ds['oswQualityCrossSpectraIm'].values.squeeze()
    re = preprocess.conv_real(cspcRe)
    im = preprocess.conv_imaginary(cspcIm)
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
        else :
            features = np.hstack([features,addon])
    #  = np.dstack([ds['cwave'].values,ds['dxdt'].values,ds['latlonSARcossin'].values,ds['todSAR'].values,ds['incidence'].values,ds['satellite'].values])
    logging.info('features ready')
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
    logging.info('test dataset ready')
    return test

def predict ( model,dataset ) :
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
    logging.info('start Hs prediction')
    # _, yhat = predict(model,test)
    yhat = predict_agrouaze(model,test)
    logging.info('prediction finished')
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
    s1_ocn_wv_ds['HsQuach'] = xarray.DataArray(data=yhat[:,0],dims=['time'])
    s1_ocn_wv_ds['HsQuach_uncertainty'] = xarray.DataArray(data=yhat[:,1],dims=['time'])
    return s1_ocn_wv_ds
