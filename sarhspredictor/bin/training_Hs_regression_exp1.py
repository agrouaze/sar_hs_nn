# Train neural network to predict significant wave height from SAR spectra.
# Train with heteroskedastic regression uncertainty estimates.
# Author: A Grouazel, June 2021
import os, sys
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Needed to avoid cudnn bug.
import numpy as np
import h5py
import datetime
import logging
import time
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint,TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D,Conv2D,GlobalMaxPooling2D,Dense,Dropout,Input,concatenate,Flatten,LSTM,Conv1D,MaxPooling1D
from sarhspredictor.config import model_IFR_replication_quach2020_sadowski_release_5feb2021_exp1
from sarhspredictor.lib.sarhs.generator import SARGenerator
from sarhspredictor.lib.sarhs.heteroskedastic import Gaussian_NLL, Gaussian_MSE
MSE_metric = tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
MAE_metric = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
MAPE_metric = tf.keras.metrics.MeanAbsolutePercentageError(name="mean_absolute_percentage_error", dtype=None)
COSI_SIMI = tf.keras.metrics.CosineSimilarity(name="cosine_similarity", dtype=None, axis=-1)
def define_model (drop_out=0.5) :
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
    x = Dropout(drop_out)(x)
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
    x = Dropout(drop_out)(x)
    ann = Model(inputs=inp,outputs=x)

    # Combine
    combinedInput = concatenate([cnn.output,ann.output])
    x = Dense(256,activation="relu")(combinedInput)
    x = Dropout(drop_out)(x)
    x = Dense(256,activation="relu",name='penultimate')(x)
    x = Dropout(drop_out)(x)
    x = Dense(2,activation="softplus",name='output')(x)
    model = Model(inputs=[cnn.input,ann.input],outputs=x)
    return model

def start_training(learning_rate=0.0001,batch_size = 128,drop_out=0.5,tblogdir=None,save_model=False,
                   hparams=None,checkPointModelSave=None):

    # Train
    if tblogdir is None:
        tblogdir = os.path.join('/home1/scratch/agrouaze/tmp/',
                            'exp_1_hs_wv_slc')
    if os.path.exists(tblogdir) is False :
        logging.info('logdir_tensorboard : %s doesnt exist',tblogdir)
        os.makedirs(tblogdir,0o0775)
    tensorBoard = TensorBoard(
        log_dir=tblogdir,
        histogram_freq=1,
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
        update_freq='epoch')

    # file_model = '/home1/scratch/agrouaze/heteroskedastic_2017_agrouaze.h5'
    if checkPointModelSave is None:
        file_model_check_point = model_IFR_replication_quach2020_sadowski_release_5feb2021_exp1  #
    else:
        file_model_check_point = checkPointModelSave
    print(file_model_check_point)
    model = define_model(drop_out=drop_out)
    model.compile(loss=Gaussian_NLL,optimizer=Adam(lr=learning_rate),metrics=[Gaussian_MSE,MAE_metric, MAPE_metric,
                                                                              COSI_SIMI,MSE_metric])
    # input dataset for the training
    file_dest2 = '/home1/scratch/agrouaze/training_quach_redo_model/aggregated_grouped_final_exp1.h5'
    # file provided to Zhengyang 20 of sept 2021:
    file_dest2 = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/aggregated_grouped_final_exp1_per_year_v21sept2021.h5'
    file_dest2 = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/D1_v2_v5oct2021.h5'
    # Dataset

    epochs = 123
    # filename = '../../data/alt/sar_hs.h5'
    # filename = '/mnt/tmp/psadow/sar/sar_hs.h5'
    filename = file_dest2
    print(file_dest2)
    train = SARGenerator(filename=filename,
                         subgroups=['2015_2016', '2017'] ,
                         batch_size=batch_size,exp=1,levelinputs='slc')
    valid = SARGenerator(filename=filename,subgroups=['2018'],batch_size=batch_size,exp=1,levelinputs='slc')
    # filename = '/mnt/tmp/psadow/sar/sar_hs.h5'
    # epochs = 25
    # train = SARGenerator(filename=filename,
    #                      subgroups=['2015_2016', '2017', '2018'], # Train on all data without early stopping.
    #                      batch_size=batch_size)

    # Callbacks
    # This LR schedule is slower than in the paper.
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.9,patience=1) # present in P;Sadoski  shared code
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=4) # conform to Quach paper
    check = ModelCheckpoint(file_model_check_point,monitor='val_loss',verbose=0,
                            save_best_only=True,save_weights_only=False,
                            mode='auto',save_freq='epoch')
    stop = EarlyStopping(monitor='val_loss',min_delta=0,patience=10,verbose=0,
                         mode='auto',baseline=None,restore_best_weights=False)
    clbks = [reduce_lr,check,stop,tensorBoard] # hp.KerasCallback(tblogdir, hparams)

    history = model.fit(train,
                        epochs=epochs,
                        validation_data=valid,
                        callbacks=clbks,
                        verbose=1)
    version_model_utput = 1
    # if save_model:
    #     outputmodel = '/home1/datawork/agrouaze/model_Hs_NN_WV_ALTIcwaveV4_regression_exp1_%s.h5' % version_model_utput
    #     model.save(outputmodel)
    logging.info('output NN model saved: %s',file_model_check_point)
    best_mse = np.min(history.history['mean_squared_error'])
    best_loss = np.min(history.history['val_loss'])
    best_mae = np.min(history.history['mean_absolute_error'])
    best_mape = np.min(history.history['mean_absolute_percentage_error'])
    best_cos = np.min(history.history['cosine_similarity'])
    return best_mse,best_loss,best_mae,best_mape,best_cos

if __name__ =='__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='exp1 Hs regression')
    parser.add_argument('--verbose',action='store_true',default=False)


    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()
    output_model = '/home1/scratch/agrouaze/test_exp1_complete_dataset_provided_to_zhengyang/keras_model_exp1_D1_v2.h5'
    start_training(checkPointModelSave=output_model)
    logging.info('elapsed time %1.1f sec',time.time()-t0)