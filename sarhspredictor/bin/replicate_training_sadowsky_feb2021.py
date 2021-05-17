"""
A Grouazel
April 2021
original notebook (where the dataset is prepared:
sarhspredictor/bin/redo_quach2020_sadowsky_model_provided_in_feb_2021_at_ifremer.ipynb
pres: https://docs.google.com/presentation/d/1hEZsDnlsKlgYC98Ha3ZF0dioWMGbhvPFlUSQuT_mw68/edit?usp=sharing

ENV : pytorchtest
machine datarmor node with GPU or only CPU, memory 25Go
pour tensorboard
(base) agrouaze@grougrou1 14:11:51 /home1/scratch/agrouaze/training_quach_redo_model conda activate pytorchtest
(pytorchtest) agrouaze@grougrou1 14:11:56 /home1/scratch/agrouaze/training_quach_redo_model tensorboard --logdir /home1/scratch/agrouaze/quach2020/20210413/
"""
# Train neural network to predict significant wave height from SAR spectra.
# Train with heteroskedastic regression uncertainty estimates.
# Author: Peter Sadowski, Dec 2020
import os, sys
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Needed to avoid cudnn bug.
import numpy as np
from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
import tensorflow as tf
tf.random.set_seed(2)
import h5py
import logging
import datetime
import time

from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Dense,Dropout,GlobalMaxPooling2D,Conv2D,MaxPooling2D,concatenate
from tensorflow.keras.models import Model

#sys.path = ['../'] + sys.path
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/')
from sarhspredictor.lib.sarhs.generator import SARGenerator
from sarhspredictor.lib.sarhs.heteroskedastic import Gaussian_NLL, Gaussian_MSE
from sarhspredictor.config import model_IFR_replication_quach2020_sadowski_release_5feb2021
# Train
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def define_model () :
    # Low-level features.
    inputs = Input(shape=(72,60,2))
    x = Conv2D(64,(3,3),activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(128,(3,3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(256,(3,3),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = GlobalMaxPooling2D()(x)
    x = Dense(256,activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)
    cnn = Model(inputs,x)

    # High-level features.
    inp = Input(shape=(32,))  # 'hsSM', 'hsWW3v2', 'hsALT', 'altID', 'target' -> dropped
    x = Dense(units=256,activation='relu')(inp)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=256,activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(units=256,activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dropout(0.5)(x)
    ann = Model(inputs=inp,outputs=x)

    # Combine
    combinedInput = concatenate([cnn.output,ann.output])
    x = Dense(256,activation="relu")(combinedInput)
    x = Dropout(0.5)(x)
    x = Dense(256,activation="relu",name='penultimate')(x)
    x = Dropout(0.5)(x)
    x = Dense(2,activation="softplus",name='output')(x)
    model = Model(inputs=[cnn.input,ann.input],outputs=x)
    return model


if __name__ =='__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='production of figures for cyclic reports CFOSAT SCAT')
    parser.add_argument('--verbose',action='store_true',default=False)
    parser.add_argument('--copytrainingdatasetlocaly',action='store_true',default=False)
    parser.add_argument('--outputModelFile',default=model_IFR_replication_quach2020_sadowski_release_5feb2021,
                        help='.h5 keras model file path to save',required=False)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    #file_model = '/home1/scratch/agrouaze/heteroskedastic_2017_agrouaze.h5'
    #file_model = model_IFR_replication_quach2020_sadowski_release_5feb2021 #
    #file_model =
    filename_out = args.outputModelFile
    filename_out = '/home1/datahome/agrouaze/sources/sentinel1/hs_total/validation_quach2020/heteroskedastic_2017_agrouaze_v2.h5'
    filename_out = '/home1/datahome/agrouaze/sources/sentinel1/hs_total/validation_quach2020/heteroskedastic_2017_agrouaze_v3.h5' #100 patience early stop
    filename_out = '/home1/datahome/agrouaze/sources/sentinel1/hs_total/validation_quach2020/heteroskedastic_2017_agrouaze_v4_%s'%datetime.datetime.today().strftime('%Y%m%d_%H%M%S.h5')
    print(filename_out)
    t0 = time.time()
    model = define_model()
    model.compile(loss=Gaussian_NLL, optimizer=Adam(lr=0.0001), metrics=[Gaussian_MSE])

    # Dataset
    batch_size = 128
    epochs = 123
    #filename = '../../data/alt/sar_hs.h5'
    #filename = '/mnt/tmp/psadow/sar/sar_hs.h5'

    filename_input = '/home1/scratch/agrouaze/training_quach_redo_model/aggregated_grouped_final.h5' # see redo_quach2020_sadowsky_model_provided_in_feb_2021_at_ifremer.ipynb for redoing this dataset
    import shutil

    logging.info('input file: %s',filename_input)
    if args.copytrainingdatasetlocaly:
        desst = os.path.join('/tmp',os.path.basename(filename_input))
        shutil.copy(filename_input,desst)
        logging.info('copy of the file on local dist : %s',desst)

    else:
        desst = filename_input
    logging.info('input file: %s',desst)
    logging.info('output file: %s',filename_out)
    train = SARGenerator(filename=desst,
                         subgroups=['2015_2016', '2017'],
                         batch_size=batch_size)
    valid = SARGenerator(filename=desst, subgroups=['2018'], batch_size=batch_size)
    # filename = '/mnt/tmp/psadow/sar/sar_hs.h5'
    # epochs = 25
    # train = SARGenerator(filename=filename,
    #                      subgroups=['2015_2016', '2017', '2018'], # Train on all data without early stopping.
    #                      batch_size=batch_size)

    # Callbacks
    logdir_tensorboard = os.path.join('/home1/scratch/agrouaze','quach2020',datetime.datetime.today().strftime('%Y%m%d'))
    if os.path.exists(logdir_tensorboard) is False:
        logging.info('logdir_tensorboard : %s doesnt exist',logdir_tensorboard)
        os.makedirs(logdir_tensorboard,0o0775)
    tensorBoard = TensorBoard(
        log_dir = logdir_tensorboard,
        histogram_freq = 0,
        batch_size = batch_size,
        write_graph = True,
        write_grads = False,
        write_images = False,
        embeddings_freq = 0,
        embeddings_layer_names = None,
        embeddings_metadata = None,
        embeddings_data = None,
        update_freq = 'epoch')
    # This LR schedule is slower than in the paper.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=1)
    check = ModelCheckpoint(filename_out, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=False,
                            mode='auto', save_freq='epoch')
    stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, #patience was 10 with Peter
                         mode='auto', baseline=None, restore_best_weights=False)
    clbks = [reduce_lr, check, stop,tensorBoard] #tensorBoard out because I suspect it slow down the process

    history = model.fit(train,
                        epochs=epochs,
                        validation_data=valid,
                        callbacks=clbks,
                        verbose=1)
    model.save(filename_out)
    logging.info('%s saved written successfully',filename_out)
    elaps = (time.time()-t0)/60.
    logging.info('done: elapsed time %1.3f min',elaps)
