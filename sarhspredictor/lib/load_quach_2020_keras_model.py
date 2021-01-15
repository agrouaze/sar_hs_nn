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
import sherpa
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from shutil import copyfile
import tensorflow as tf

from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
import tensorflow.keras as keras
import importlib
print('keras',keras.__version__)
print('tensorflow',tf.__version__)
print()
import keras.backend as K
CURDIRSCRIPT = os.path.dirname(__file__)
# def gaussian_nll ( ytrue,ypreds ) :
#     """
#     Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
#     This implementation implies diagonal covariance matrix.
#
#     Parameters
#     ----------
#     ytrue: tf.tensor of shape [n_samples, n_dims]
#         ground truth values
#     ypreds: tf.tensor of shape [n_samples, n_dims*2]
#         predicted mu and logsigma values (e.g. by your neural network)
#
#     Returns
#     -------
#     neg_log_likelihood: float
#         negative loglikelihood averaged over samples
#
#     This loss can then be used as a target loss for any keras model, e.g.:
#         model.compile(loss=gaussian_nll, optimizer='Adam')
#
#     """
#
#     n_dims = int(int(ypreds.shape[1]) / 2)
#     mu = ypreds[:,0 :n_dims]
#     logsigma = ypreds[:,n_dims :]
#
#     mse = -0.5 * K.sum(K.square((ytrue - mu) / K.exp(logsigma)),axis=1)
#     sigma_trace = -K.sum(logsigma,axis=1)
#     log2pi = -0.5 * n_dims * np.log(2 * np.pi)
#
#     log_likelihood = mse + sigma_trace + log2pi
#
#     return K.mean(-log_likelihood)


from sarhs.heteroskedastic import Gaussian_NLL, Gaussian_MSE
# try :
#     import keras_extras
#     importlib.reload(keras_extras.losses.dirichlet)
#     from keras_extras.losses.dirichlet import Gaussian_NLL,Gaussian_MSE
# except :  # agrouaze change: I dont know which lib is the keras_extras
#     Gaussian_MSE = tf.keras.losses.MeanSquaredError()
#     Gaussian_NLL = gaussian_nll


def load_quach2020_model_basic():
    """
    file provided by Justin in December 2020
    :return:
    """
    file_model = os.path.join(CURDIRSCRIPT,'model_45.h5')
    logging.info('file_model: %s',file_model)
    model_base = load_model(file_model)
    return model_base


def load_quach2020_model():
    """
    version with STD uncertainty better ???
    file provided by Justin in December 2020
    :return:
    """
    file_model = os.path.join(CURDIRSCRIPT,'model_45.h5')
    logging.info('file_model: %s',file_model)
    model_base = load_model(file_model)
    # Add back output that predicts uncertainty.
    base_inputs = model_base.input
    base_penultimate = model_base.get_layer('dense_7').output
    base_output = model_base.output
    stdtuned_path = os.path.join(CURDIRSCRIPT,'model_45_std_tuned.h5')
    logging.info('stdtuned_path %s',stdtuned_path)
    model_std = load_model(stdtuned_path,#'./model_45_std_tuned.h5',
                           custom_objects={'Gaussian_NLL' : Gaussian_NLL,'Gaussian_MSE' : Gaussian_MSE})
    x = model_std.get_layer('std_hidden')(base_penultimate)
    std_output = model_std.get_layer('std_output')(x)
    output = concatenate([base_output,std_output],axis=-1)
    print('output',output)
    print(dir(output))
    model = Model(inputs=base_inputs,outputs=output)
    # Compile and save.
    opt = Adam(lr=0.0001)
    model.compile(loss=Gaussian_NLL,optimizer=opt,metrics=[Gaussian_MSE])
    logging.info('model Quach 2020 loaded')
    return model

def load_quach2020_model_v2():
    """
    based on the example notebook provided by P. Sadowsky: predict.ipynb
    :return:
    """
    file_model = '/home1/datahome/agrouaze/git/SAR-Wave-Height/models/heteroskedastic_2017.h5'
    custom_objects = {'Gaussian_NLL':Gaussian_NLL, 'Gaussian_MSE':Gaussian_MSE}
    model = load_model(file_model, custom_objects=custom_objects)
    return model


def load_quach2020_model_45_std_tuned():
    stdtuned_path = os.path.join(CURDIRSCRIPT,'model_45_std_tuned.h5')
    custom_objects = {'Gaussian_NLL' : Gaussian_NLL,'Gaussian_MSE' : Gaussian_MSE}
    model = load_model(stdtuned_path,custom_objects=custom_objects)
    return model


def load_like_in_notebook_train_uncertainty_with_existing():
    """
    Un problem je nai pas le dataset "valid" en hdf5.
    :return:
    """
    #file_model = './models/model_45.h5'
    file_model = os.path.join(CURDIRSCRIPT,'model_45.h5')
    model_base = load_model(file_model)
    # Fine tune.
    opt = Adam(lr=0.00001)
    model_base.compile(loss='mae', optimizer=opt, metrics=['mae', 'mse'])
    history = model_base.fit(valid, epochs=5, verbose= 1 if INTERACTIVE else 2)
    # Add back output that predicts uncertainty.
    base_inputs = model_base.input
    base_penultimate = model_base.get_layer('dense_7').output
    base_output = model_base.output
    model_std = load_model('./model_45_std.h5', custom_objects={'Gaussian_NLL':Gaussian_NLL, 'Gaussian_MSE': Gaussian_MSE})
    x = model_std.get_layer('std_hidden')(base_penultimate)
    std_output = model_std.get_layer('std_output')(x)
    output = concatenate([base_output, std_output], axis=-1)
    model = Model(inputs=base_inputs, outputs=output)
    # Compile and save.
    opt = Adam(lr=0.0001)
    model.compile(loss=Gaussian_NLL, optimizer=opt, metrics=[Gaussian_MSE])
    return model

POSSIBLES_MODELS = {
    'heteroskedastic_2017.h5' : load_quach2020_model_v2,
    'justin_std' : load_quach2020_model,
    'justin_basic' : load_quach2020_model_basic,
    'only_std' : load_quach2020_model_45_std_tuned,
}
