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
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
import tensorflow.keras as keras
from sarhspredictor.config import model45path,model45stdpath,model_heteroskedastic_2017
print('keras',keras.__version__)
print('tensorflow',tf.__version__)
print()
from sarhspredictor.lib.sarhs.heteroskedastic import Gaussian_NLL, Gaussian_MSE
CURDIRSCRIPT = os.path.dirname(__file__)

def load_quach2020_model_basic():
    """
    file provided by Justin in December 2020
    it only predicts Hs and not the Hs_std
    :return:
    """
    #file_model = os.path.join(CURDIRSCRIPT,'model_45.h5')
    file_model = model45path
    logging.info('file_model: %s',file_model)
    model_base = load_model(file_model)
    return model_base


def load_quach2020_model():
    """
    version with STD uncertainty inspired from https://github.com/hawaii-ai/SAR-Wave-Height/blob/master/notebooks/train_uncertainty_with_existing.ipynb
    file provided by Justin in December 2020
    :return:
    """
    #file_model = os.path.join(CURDIRSCRIPT,'model_45.h5')
    file_model = model45path
    logging.info('file_model: %s',file_model)
    model_base = load_model(file_model)
    # Add back output that predicts uncertainty.
    base_inputs = model_base.input
    base_penultimate = model_base.get_layer('dense_7').output
    base_output = model_base.output
    #stdtuned_path = os.path.join(CURDIRSCRIPT,'model_45_std_tuned.h5')
    stdtuned_path = model45stdpath
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
    #file_model = '/home1/datahome/agrouaze/git/SAR-Wave-Height/models/heteroskedastic_2017.h5'
    file_model = model_heteroskedastic_2017
    custom_objects = {'Gaussian_NLL':Gaussian_NLL, 'Gaussian_MSE':Gaussian_MSE}
    model = load_model(file_model, custom_objects=custom_objects)
    return model


def load_quach2020_model_45_std_tuned():
    """
    temptative to load only the model_45_std_tuned.h5 model by itself -> validation tests not conclusive
    :return:
    """
    stdtuned_path = os.path.join(CURDIRSCRIPT,'model_45_std_tuned.h5')
    custom_objects = {'Gaussian_NLL' : Gaussian_NLL,'Gaussian_MSE' : Gaussian_MSE}
    model = load_model(stdtuned_path,custom_objects=custom_objects)
    return model


# def load_like_in_notebook_train_uncertainty_with_existing():
#     """
#     problem I don't have the  "valid" dataset in hdf5.for now.
#     :return:
#     """
#     #file_model = './models/model_45.h5'
#     #file_model = os.path.join(CURDIRSCRIPT,'model_45.h5')
#     file_model = model45path
#     model_base = load_model(file_model)
#     # Fine tune.
#     opt = Adam(lr=0.00001)
#     model_base.compile(loss='mae', optimizer=opt, metrics=['mae', 'mse'])
#     history = model_base.fit(valid, epochs=5, verbose= 1 if INTERACTIVE else 2)
#     # Add back output that predicts uncertainty.
#     base_inputs = model_base.input
#     base_penultimate = model_base.get_layer('dense_7').output
#     base_output = model_base.output
#     model_std = load_model(model45stdpath, custom_objects={'Gaussian_NLL':Gaussian_NLL, 'Gaussian_MSE': Gaussian_MSE})
#     x = model_std.get_layer('std_hidden')(base_penultimate)
#     std_output = model_std.get_layer('std_output')(x)
#     output = concatenate([base_output, std_output], axis=-1)
#     model = Model(inputs=base_inputs, outputs=output)
#     # Compile and save.
#     opt = Adam(lr=0.0001)
#     model.compile(loss=Gaussian_NLL, optimizer=opt, metrics=[Gaussian_MSE])
#     return model

POSSIBLES_MODELS = {
    'heteroskedastic_2017.h5' : load_quach2020_model_v2,
    'justin_std' : load_quach2020_model,
    'justin_basic' : load_quach2020_model_basic,
    'only_std' : load_quach2020_model_45_std_tuned,
}
