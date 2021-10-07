"""
need to do groups in the fat aggregated training dataset( alti+SAr Cwave params)
copy paste from https://github.com/hawaii-ai/SAR-Wave-Height/blob/master/scripts/create_dataset.ipynb
April 2021
A Grouazel
env used so far: jupyter-tensorflow-ric -> /appli/conda-env/jupyterhub-tensorflow/bin/python

"""
import numpy as np
import glob
import h5py
import pandas as pd
#from tqdm import tqdm
import os, sys
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn')
import logging
from sarhspredictor.lib.sarhs import preprocess


# groups = {'2017':[2017], '2018':[2018]}
file_src = '/home/psadow/lts/preserve/stopa/sar_hs/data/alt/aggregated.h5'
file_dest = '/home/psadow/lts/preserve/stopa/sar_hs/data/alt/sar_hs.h5'

# For competition, grab data for 2019.
# groups = {}
# for isat in range(2):
#     for year in [2019]:
#         for imonth in range(12):
#             sat = 'A' if isat==1 else 'B'
#             month = imonth+1
#             name = f'S1{sat}_{year}{month:02d}S'
#             groups[name] = (isat, year, month)
# file_src = '/home/psadow/lts/preserve/stopa/sar_hs/data/alt/aggregated_2019.h5'
# file_dest = '/home/psadow/lts/preserve/stopa/sar_hs/data/alt/sar_hs_2019.h5'
def split_aggregated_ds(file_src,file_dest):
    """

    :param file_src:
    :param file_dest:
    :return:
    """
    groups = {'2015_2016' : [2015,2016],'2017' : [2017],'2018' : [2018]}
    # Print fields of source file.
    with h5py.File(file_src,'r') as f :
        for k in [k for k in f.keys()] :
            print(f'{k}: {f[k].dtype}')

    # Create h5.
    with h5py.File(file_src,'r') as fs,h5py.File(file_dest,'w') as fd :
        for group_name,years in groups.items() :
            grp = fd.create_group(group_name)

            # Find examples of the specified years.
            indices = np.zeros_like(fs['year'][:],dtype='bool')
            for year in years :
                indices = np.logical_or(fs['year'][:] == year,indices)
            # Find examples that don't have nans.
            indices[np.any(np.isnan(fs['py_S'][:]),axis=1)] = 0
            indices[np.isnan(fs['sigma0'][:])] = 0
            indices[np.isnan(fs['normalizedVariance'][:])] = 0
            # Done
            num_examples = indices.sum()
            print(f'Found {num_examples} events from years: ',years)

            # Write data from this year.
            # print(fs['year'][indices].shape)
            grp.create_dataset('year',data=fs['year'][indices])

            # Get 22 CWAVE features.
            cwave = np.hstack([fs['py_S'][indices,...],fs['sigma0'][indices].reshape(-1,1),
                               fs['normalizedVariance'][indices].reshape(-1,1)])
            cwave = preprocess.conv_cwave(cwave)  # Remove extrema, then standardize with hardcoded mean,vars.
            grp.create_dataset('cwave',data=cwave)

            # Additional features.
            dx = preprocess.conv_dx(fs['dx'][indices])
            dt = preprocess.conv_dt(fs['dt'][indices])
            grp.create_dataset('dxdt',data=np.column_stack([dx,dt]))

            latSAR = fs['latSAR'][indices]
            lonSAR = fs['lonSAR'][indices]
            latSARcossin = preprocess.conv_position(latSAR)  # Gets cos and sin
            lonSARcossin = preprocess.conv_position(lonSAR)
            grp.create_dataset('latlonSAR',data=np.column_stack([latSAR,lonSAR]))
            grp.create_dataset('latlonSARcossin',data=np.hstack([latSARcossin,lonSARcossin]))
            logging.info('taille timeSAR %s indices : %s elem : %s',fs['timeSAR'][:].shape,indices.shape,indices.sum())
            timeSAR = fs['timeSAR'][:].squeeze()[indices]
            todSAR = preprocess.conv_time(timeSAR)
            grp.create_dataset('timeSAR',data=timeSAR,shape=(timeSAR.shape[0],1))
            grp.create_dataset('todSAR',data=todSAR,shape=(todSAR.shape[0],1))

            incidence = preprocess.conv_incidence(fs['incidenceAngle'][indices])  # Separates into 2 var.
            grp.create_dataset('incidence',data=incidence)

            satellite = fs['satellite'][indices]
            grp.create_dataset('satellite',data=satellite,shape=(satellite.shape[0],1))

            # Altimeter
            hsALT = fs['hsALT'][:].squeeze()[indices]
            grp.create_dataset('hsALT',data=hsALT,shape=(hsALT.shape[0],1))

            # Get spectral data.
            x = np.stack((preprocess.conv_real(fs['cspcRe'][indices,...]),
                          preprocess.conv_imaginary(fs['cspcIm'][indices,...]),
                          ),
                         axis=3)
            grp.create_dataset('spectrum',data=x)
            print(f'Done with {years}')
    print('Done')

def split_aggregated_ds_v2(file_src,file_dest,test2015=False,exp_id=1):
    """
    ma version pcq je pense que ca nest aps une bonne idee de refaire les normalization deja faite dans le training dataset
    :param file_src:
    :param file_dest:
    :return:
    """
    logging.info('start splitting into groups')
    groups = {'2015_2016' : [2015,2016],'2017' : [2017],'2018' : [2018]}
    #groups = {'group_1':[1,2,3,4,5,6],'group_2':[7,8,9],'group_3':[10,11,12]} # for dev on exp1 I only have 2015 for now
    # Print fields of source file.
    with h5py.File(file_src,'r') as f :
        for k in [k for k in f.keys()] :
            logging.debug('k',k)
            #print(f'{k}: {f[k].dtype}')
    logging.info('start creating the final .h5 file')
    # Create h5.
    with h5py.File(file_src,'r') as fs,h5py.File(file_dest,'w') as fd :
        for group_name,years in groups.items() :
        #for group_name,months in groups.items() :
            logging.info('group_name: %s years: %s',group_name,years)
            #logging.info('group_name: %s months: %s',group_name,months)
            grp = fd.create_group(group_name)

            if test2015:
                months =[] #to remove if test only on 2015 months
                # Find examples of the specified months.
                indices = np.zeros_like(fs['month'][:],dtype='bool')
                logging.info('month val :%s %s',fs['month'].shape,fs['month'][0].dtype)
                for month in months :
                    indices = np.logical_or(fs['month'][:] == month,indices)
                    logging.info('indices %s %s %s',month,indices.shape,indices.sum())
            else:
                #years = []# to remove for dataset on many years
                # Find examples of the specified years.
                indices = np.zeros_like(fs['year'][:],dtype='bool')
                for year in years :
                    indices = np.logical_or(fs['year'][:] == year,indices)
                    logging.info('indices %s %s %s',year,indices.shape,indices.sum())
            # Find examples that don't have nans.
            indices[np.any(np.isnan(fs['py_S'][:]),axis=1)] = 0
            indices[np.isnan(fs['sigma0'][:])] = 0
            indices[np.isnan(fs['normalizedVariance'][:])] = 0
            #I add other features because in 2017 it crash the learning (agrouaze June 2021)
            logging.info('assert that no NaN in high level features!! indices before : %s',indices.sum())
            indices[np.isnan(fs['latSAR'][:])] = 0
            logging.info('indices sum after lat : %s',indices.sum())
            indices[np.isnan(fs['lonSAR'][:])] = 0
            logging.info('indices sum after lon : %s',indices.sum())
            indices[np.isnan(fs['incidenceAngle'][:])] = 0
            logging.info('indices sum after inc : %s',indices.sum())
            indices[np.isnan(fs['dx'][:])] = 0
            logging.info('indices sum after dx : %s',indices.sum())
            indices[np.isnan(fs['dt'][:])] = 0
            logging.info('indices sum after dt : %s',indices.sum())
            indices[np.isnan(fs['timeSAR'][:].squeeze())] = 0
            logging.info('indices sum after time : %s',indices.sum())
            indices[np.isnan(fs['todSAR'][:].squeeze())] = 0
            logging.info('indices sum after tod : %s',indices.sum())
            indices[np.isnan(fs['satellite'][:].squeeze())] = 0
            logging.info('indices sum after sat : %s',indices.sum())
            indices[np.isnan(fs['hsALT'][:].squeeze())] = 0
            logging.info('indices sum after hsalt : %s',indices.sum())
            logging.info('cspcIm_slc shape : %s',fs['cspcIm_slc'][:].shape)
            indices[np.any(np.any(np.isnan(fs['cspcIm_slc'][:]),axis=2),axis=1)] = 0
            logging.info('indices sum after Im : %s',indices.sum())
            indices[np.any(np.any(np.isnan(fs['cspcRe_slc'][:]),axis=2),axis=1)] = 0
            logging.info('indices sum after Re : %s',indices.sum())
            # Done
            num_examples = indices.sum()
            logging.info('Found %s events from years: %s ',num_examples,years)
            #print(f'Found {num_examples} events from months: ',months)

            # Write data from this year.
            # print(fs['year'][indices].shape)
            grp.create_dataset('year',data=fs['year'][indices])

            # Get 22 CWAVE features.
            #cwave = np.hstack([fs['py_S'][indices,...],fs['sigma0'][indices].reshape(-1,1),
            #                   fs['normalizedVariance'][indices].reshape(-1,1)])
            #cwave = preprocess.conv_cwave(cwave)  # Remove extrema, then standardize with hardcoded mean,vars.
            cwave_ocn = fs['cwave_ocn'][indices,...]
            grp.create_dataset('cwave_ocn',data=cwave_ocn)

            cwave_slc = fs['cwave'][indices,...]
            grp.create_dataset('cwave_slc',data=cwave_slc)

            # Additional features.
            dx = preprocess.conv_dx(fs['dx'][indices]) #I keep the normalisation here for dx and dt
            dt = preprocess.conv_dt(fs['dt'][indices])
            grp.create_dataset('dxdt',data=np.column_stack([dx,dt]))

            latSAR = fs['latSAR'][indices]
            lonSAR = fs['lonSAR'][indices]
            latSARcossin = preprocess.conv_position(latSAR)  # Gets cos and sin
            lonSARcossin = preprocess.conv_position(lonSAR)
            grp.create_dataset('latlonSAR',data=np.column_stack([latSAR,lonSAR]))
            grp.create_dataset('latlonSARcossin',data=np.hstack([latSARcossin,lonSARcossin]))
            #print('timeSAR',fs['timeSAR'].shape)
            timeSAR = fs['timeSAR'][:].squeeze()[indices]
            #todSAR = preprocess.conv_time(timeSAR)
            todSAR = fs['todSAR'][:].squeeze()[indices]
            grp.create_dataset('timeSAR',data=timeSAR,shape=(timeSAR.shape[0],1))
            grp.create_dataset('todSAR',data=todSAR,shape=(todSAR.shape[0],1))

            incidence = preprocess.conv_incidence(fs['incidenceAngle'][indices])  # Separates into 2 var.
            grp.create_dataset('incidence',data=incidence)

            satellite = fs['satellite'][indices]
            grp.create_dataset('satellite',data=satellite,shape=(satellite.shape[0],1))

            # Altimeter
            hsALT = fs['hsALT'][:].squeeze()[indices]
            grp.create_dataset('hsALT',data=hsALT,shape=(hsALT.shape[0],1))

            # Get spectral data.
            logging.info('fs[cspcRe_slc] : %s',fs['cspcRe_slc'].shape)
            tmpRe_slc = fs['cspcRe_slc'][indices,...].squeeze()
            #tmpRe = np.swapaxes(tmpRe,1,2)
            tmpRe_slc = np.swapaxes(tmpRe_slc,1,2)
            tmpIm_slc = fs['cspcIm_slc'][indices,...].squeeze()
            tmpIm_slc = np.swapaxes(tmpIm_slc,1,2)
            logging.info('tmpIm_slc : %s',tmpIm_slc.shape)
            x = np.stack((preprocess.conv_real(tmpRe_slc,exp_id=1),
                          preprocess.conv_imaginary(tmpIm_slc,exp_id=1),
                          ),
                         axis=3)
            grp.create_dataset('spectrum_slc',data=x)

            # Get spectral data.
            logging.info('fs[cspcRe_ocn] : %s',fs['cspcRe_ocn'].shape)
            tmpRe = fs['cspcRe_ocn'][indices,...].squeeze()
            # tmpRe = np.swapaxes(tmpRe,1,2)
            tmpIm = fs['cspcIm_ocn'][indices,...].squeeze()
            # tmpIm = np.swapaxes(tmpIm,1,2)
            logging.info('tmpIm : %s',tmpIm.shape)
            x_ocn = np.stack((preprocess.conv_real(tmpRe,exp_id=None),
                          preprocess.conv_imaginary(tmpIm,exp_id=None),
                          ),
                         axis=3)
            grp.create_dataset('spectrum_ocn',data=x_ocn)
            #print(f'Done with {months}')
            logging.info('Done with %s',years)
    logging.info('Done')

def split_aggregated_ds_v3(file_src,file_dest,exp_id=1):
    """
    ma version pcq je pense que ca nest aps une bonne idee de refaire les normalization deja faite dans le training dataset
    + v3: splitting nto depending on dates but always 80% train 20% valid
    :param file_src:
    :param file_dest:
    :return:
    """
    logging.info('start splitting into groups')
    #groups = {'2015_2016' : [2015,2016],'2017' : [2017],'2018' : [2018]}
    groups = {'training_dataset':80,'validation_dataset':20}
    #groups = {'group_1':[1,2,3,4,5,6],'group_2':[7,8,9],'group_3':[10,11,12]} # for dev on exp1 I only have 2015 for now
    # Print fields of source file.
    with h5py.File(file_src,'r') as f :
        for k in [k for k in f.keys()] :
            logging.debug('k',k)
            #print(f'{k}: {f[k].dtype}')
    logging.info('start creating the final .h5 file')
    # Create h5.
    with h5py.File(file_src,'r') as fs,h5py.File(file_dest,'w') as fd :
        total_size = len(fs['month'][:])
        for group_name,pct_data in groups.items() :
        #for group_name,months in groups.items() :
            logging.info('group_name: %s pct_data: %s',group_name,pct_data)
            #logging.info('group_name: %s months: %s',group_name,months)
            grp = fd.create_group(group_name)


            # Find examples of the specified years.
            indices = np.zeros_like(fs['year'][:],dtype='bool')
            indices_val  = np.arange(total_size)
            if group_name=='training_dataset':
                indices = np.logical_or(indices_val<=int(total_size*groups['training_dataset']/100.),indices)
            else:
                indices = np.logical_or(indices_val > int(total_size * groups['training_dataset']/100.),indices)
            logging.info('indices %s %s %s',group_name,indices.shape,indices.sum())
            # Find examples that don't have nans.
            indices[np.any(np.isnan(fs['py_S'][:]),axis=1)] = 0
            indices[np.isnan(fs['sigma0'][:])] = 0
            indices[np.isnan(fs['normalizedVariance'][:])] = 0
            #I add other features because in 2017 it crash the learning (agrouaze June 2021)
            logging.info('assert that no NaN in high level features!! indices before : %s',indices.sum())
            indices[np.isnan(fs['latSAR'][:])] = 0
            logging.info('indices sum after lat : %s',indices.sum())
            indices[np.isnan(fs['lonSAR'][:])] = 0
            logging.info('indices sum after lon : %s',indices.sum())
            indices[np.isnan(fs['incidenceAngle'][:])] = 0
            logging.info('indices sum after inc : %s',indices.sum())
            indices[np.isnan(fs['dx'][:])] = 0
            logging.info('indices sum after dx : %s',indices.sum())
            indices[np.isnan(fs['dt'][:])] = 0
            logging.info('indices sum after dt : %s',indices.sum())
            indices[np.isnan(fs['timeSAR'][:].squeeze())] = 0
            logging.info('indices sum after time : %s',indices.sum())
            indices[np.isnan(fs['todSAR'][:].squeeze())] = 0
            logging.info('indices sum after tod : %s',indices.sum())
            indices[np.isnan(fs['satellite'][:].squeeze())] = 0
            logging.info('indices sum after sat : %s',indices.sum())
            indices[np.isnan(fs['hsALT'][:].squeeze())] = 0
            logging.info('indices sum after hsalt : %s',indices.sum())
            logging.info('cspcIm_ocn shape : %s',fs['cspcIm_ocn'][:].shape)
            indices[np.any(np.any(np.isnan(fs['cspcIm_ocn'][:]),axis=2),axis=1)] = 0
            logging.info('indices sum after Im : %s',indices.sum())
            indices[np.any(np.any(np.isnan(fs['cspcRe_ocn'][:]),axis=2),axis=1)] = 0
            logging.info('indices sum after Re : %s',indices.sum())
            logging.info('cspcIm_slc shape : %s',fs['cspcIm_slc'][:].shape)
            indices[np.any(np.any(np.isnan(fs['cspcIm_slc'][:]),axis=2),axis=1)] = 0
            logging.info('indices sum after Im slc : %s',indices.sum())
            indices[np.any(np.any(np.isnan(fs['cspcRe_slc'][:]),axis=2),axis=1)] = 0
            logging.info('indices sum after Re slc : %s',indices.sum())
            # Done
            num_examples = indices.sum()
            logging.info('Found %s events from group: %s ',num_examples,group_name)
            #print(f'Found {num_examples} events from months: ',months)

            # Write data from this year.
            # print(fs['year'][indices].shape)
            grp.create_dataset('year',data=fs['year'][indices])

            # Get 22 CWAVE features.
            #cwave = np.hstack([fs['py_S'][indices,...],fs['sigma0'][indices].reshape(-1,1),
            #                   fs['normalizedVariance'][indices].reshape(-1,1)])
            #cwave = preprocess.conv_cwave(cwave)  # Remove extrema, then standardize with hardcoded mean,vars.
            cwave = fs['cwave'][indices,...]
            grp.create_dataset('cwave_slc',data=cwave)

            cwave_ocn = fs['cwave_ocn'][indices,...]
            grp.create_dataset('cwave_ocn',data=cwave_ocn)

            # Additional features.
            dx = preprocess.conv_dx(fs['dx'][indices]) #I keep the normalisation here for dx and dt
            dt = preprocess.conv_dt(fs['dt'][indices])
            grp.create_dataset('dxdt',data=np.column_stack([dx,dt]))

            latSAR = fs['latSAR'][indices]
            lonSAR = fs['lonSAR'][indices]
            latSARcossin = preprocess.conv_position(latSAR)  # Gets cos and sin
            lonSARcossin = preprocess.conv_position(lonSAR)
            grp.create_dataset('latlonSAR',data=np.column_stack([latSAR,lonSAR]))
            grp.create_dataset('latlonSARcossin',data=np.hstack([latSARcossin,lonSARcossin]))
            #print('timeSAR',fs['timeSAR'].shape)
            timeSAR = fs['timeSAR'][:].squeeze()[indices]
            #todSAR = preprocess.conv_time(timeSAR)
            todSAR = fs['todSAR'][:].squeeze()[indices]
            grp.create_dataset('timeSAR',data=timeSAR,shape=(timeSAR.shape[0],1))
            grp.create_dataset('todSAR',data=todSAR,shape=(todSAR.shape[0],1))

            incidence = preprocess.conv_incidence(fs['incidenceAngle'][indices])  # Separates into 2 var.
            grp.create_dataset('incidence',data=incidence)

            satellite = fs['satellite'][indices]
            grp.create_dataset('satellite',data=satellite,shape=(satellite.shape[0],1))

            # Altimeter
            hsALT = fs['hsALT'][:].squeeze()[indices]
            grp.create_dataset('hsALT',data=hsALT,shape=(hsALT.shape[0],1))

            # Get spectral data.
            logging.info('fs[cspcRe] : %s',fs['cspcRe_ocn'].shape)
            tmpRe_ocn = fs['cspcRe_ocn'][indices,...].squeeze()
            #tmpRe = np.swapaxes(tmpRe,1,2)
            tmpIm_ocn = fs['cspcIm_ocn'][indices,...].squeeze()
            #tmpIm = np.swapaxes(tmpIm,1,2)
            logging.info('tmpIm : %s',tmpIm_ocn.shape)
            x_ocn = np.stack((preprocess.conv_real(tmpRe_ocn,exp_id=None), #I set None for exp_id since foe the OCN spectra the preprocessing is the same
                          preprocess.conv_imaginary(tmpIm_ocn,exp_id=None),
                          ),
                         axis=3)
            grp.create_dataset('spectrum_ocn',data=x_ocn)

            # Get spectral data. SLC
            logging.info('fs[cspcRe_slc] : %s',fs['cspcRe_slc'].shape)
            tmpRe_slc = fs['cspcRe_slc'][indices,...].squeeze()
            tmpRe_slc = np.swapaxes(tmpRe_slc,1,2)
            tmpIm_slc = fs['cspcIm_slc'][indices,...].squeeze()
            tmpIm_slc = np.swapaxes(tmpIm_slc,1,2)
            logging.info('tmpIm_slc : %s',tmpIm_slc.shape)
            x_slc = np.stack((preprocess.conv_real(tmpRe_slc,exp_id=exp_id),
                              preprocess.conv_imaginary(tmpIm_slc,exp_id=exp_id),
                              ),
                             axis=3)
            grp.create_dataset('spectrum_slc',data=x_slc)
            #print(f'Done with {months}')
            logging.info('Done with %s',group_name)
    try:
        fd.close()
    except:
        logging.info('tried to close file dest handler unsuccessfuly (may be already closed)')
    logging.info('Done')

if __name__ =='__main__':
    # the training dataset must be separated into sub groups
    # long long task (about 30min) lots of memory
    import logging
    import time
    import argparse
    tinit = time.time()
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--verbose',action='store_true',default=False)
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    import time
    file_src2 = os.path.join('/home1/scratch/agrouaze/training_quach_redo_model/exp1',"aggregated_test_toto.h5")
    print('source ',file_src2)
    file_dest2 = '/home1/scratch/agrouaze/training_quach_redo_model/aggregated_grouped_final_exp1.h5'
    file_dest2 = '/home1/scratch/agrouaze/training_quach_redo_model/aggregated_grouped_final_exp1_per_year.h5'
    file_dest2 = '/home1/scratch/agrouaze/training_quach_redo_model/aggregated_grouped_final_exp1_per_year_v21sept2021.h5'
    file_dest2 = '/home1/scratch/agrouaze/training_quach_redo_model/aggregated_grouped_final_exp1_per_year_v5oct2021.h5'
    file_dest2 = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/D1_v2_v5oct2021.h5'
    if os.path.exists(file_dest2) :
        os.remove(file_dest2)
    t0 = time.time()
    split_aggregated_ds_v2(file_src2,file_dest2,test2015=False,exp_id=1)
    # split_aggregated_into_groups.split_aggregated_ds_v3(file_src2,file_dest2,exp_id=1) # split 80% 20%
    logging.info('done in %1.3f seconds',time.time() - t0)