"""
9 June 2021
 A Grouazel
 test multi experiment training using hyperparameters
"""
from tensorboard.plugins.hparams import api as hp
import os
import time
import logging
import tensorflow as tf
from training_Hs_regression_exp1 import start_training
if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser(description='training NN hyper Parameters')
    parser.add_argument('--verbose',action='store_true',default=False)
    args = parser.parse_args()
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    t1 = time.time()
    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
    HP_LEARNING_RATE = hp.HParam('learning_rate',  hp.Discrete([1e-2, 1e-3, 1e-4]))
    HP_BATCH_SIZE = hp.HParam('batch_size',hp.Discrete([64,128,256,512,1024]))
    results_path = '/home1/scratch/agrouaze/exp_1_hp_v2/'
    results_path = '/home1/scratch/agrouaze/exp_1_hp_v2_7oct21/'
    # Launch testing session
    session_num = 0
    for learning_rate in HP_LEARNING_RATE.domain.values:
        for batch_size in HP_BATCH_SIZE.domain.values:
            #for optimizer in HP_OPTIMIZER.domain.values:
            session_num += 1
            hparams = {
               # HP_OPTIMIZER: optimizer,
                HP_LEARNING_RATE: learning_rate,
                HP_BATCH_SIZE: batch_size
            }

            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            path_logs_tensorboard = os.path.join(results_path ,"hparam_tuning" , run_name)
            checkPointModelSave = os.path.join(results_path,'hparam_tuning',run_name,'checkpoint_model_save.h5')
            #val_loss = train_test_model(results_path + 'CMB4_V1/hparam_tuning/' + run_name, hparams, input_size)
            best_mse,best_loss,best_mae,best_mape,best_cos = start_training(learning_rate=learning_rate,
                                                            batch_size = batch_size,drop_out=0.5,
                                                            tblogdir=path_logs_tensorboard,save_model=False,hparams=hparams,
                                                            checkPointModelSave=checkPointModelSave)
            with tf.summary.create_file_writer(path_logs_tensorboard).as_default():
                tf.summary.scalar('Best val. loss', best_loss, step=1)
                tf.summary.scalar('Best mse',best_mse,step=1)
                tf.summary.scalar('Best MAE',best_mae,step=1)
                tf.summary.scalar('Best MAPE',best_mape,step=1)
                tf.summary.scalar('Best cosinus proximity',best_cos,step=1)
    logging.info('end of the hyper params run for training')