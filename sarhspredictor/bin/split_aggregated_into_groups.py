"""
need to do groups in the fat aggregated training dataset( alti+SAr Cwave params)
copy paste from https://github.com/hawaii-ai/SAR-Wave-Height/blob/master/scripts/create_dataset.ipynb
April 2021
A Grouazel

"""
import numpy as np
import glob
import h5py
import pandas as pd
#from tqdm import tqdm
import os, sys
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
            logging.info('cspcIm shape : %s',fs['cspcIm'][:].shape)
            indices[np.any(np.any(np.isnan(fs['cspcIm'][:]),axis=2),axis=1)] = 0
            logging.info('indices sum after Im : %s',indices.sum())
            indices[np.any(np.any(np.isnan(fs['cspcRe'][:]),axis=2),axis=1)] = 0
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
            cwave = fs['cwave'][indices,...]
            grp.create_dataset('cwave',data=cwave)

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
            logging.info('fs[cspcRe] : %s',fs['cspcRe'].shape)
            tmpRe = fs['cspcRe'][indices,...].squeeze()
            #tmpRe = np.swapaxes(tmpRe,1,2)
            tmpIm = fs['cspcIm'][indices,...].squeeze()
            #tmpIm = np.swapaxes(tmpIm,1,2)
            logging.info('tmpIm : %s',tmpIm.shape)
            x = np.stack((preprocess.conv_real(tmpRe,exp_id=1),
                          preprocess.conv_imaginary(tmpIm,exp_id=1),
                          ),
                         axis=3)
            grp.create_dataset('spectrum',data=x)
            #print(f'Done with {months}')
            logging.info('Done with %s',years)
    logging.info('Done')