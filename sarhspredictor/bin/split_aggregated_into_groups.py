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
from tqdm import tqdm
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

            timeSAR = fs['timeSAR'][indices]
            todSAR = preprocess.conv_time(timeSAR)
            grp.create_dataset('timeSAR',data=timeSAR,shape=(timeSAR.shape[0],1))
            grp.create_dataset('todSAR',data=todSAR,shape=(todSAR.shape[0],1))

            incidence = preprocess.conv_incidence(fs['incidenceAngle'][indices])  # Separates into 2 var.
            grp.create_dataset('incidence',data=incidence)

            satellite = fs['satellite'][indices]
            grp.create_dataset('satellite',data=satellite,shape=(satellite.shape[0],1))

            # Altimeter
            hsALT = fs['hsALT'][indices]
            grp.create_dataset('hsALT',data=hsALT,shape=(hsALT.shape[0],1))

            # Get spectral data.
            x = np.stack((preprocess.conv_real(fs['cspcRe'][indices,...]),
                          preprocess.conv_imaginary(fs['cspcIm'][indices,...]),
                          ),
                         axis=3)
            grp.create_dataset('spectrum',data=x)
            print(f'Done with {years}')
    print('Done')

def split_aggregated_ds_v2(file_src,file_dest,test2015=False):
    """
    ma version pcq je pense que ca nest aps une bonne idee de refaire les normalization deja faite dans le training dataset
    :param file_src:
    :param file_dest:
    :return:
    """
    #groups = {'2015_2016' : [2015,2016],'2017' : [2017],'2018' : [2018]}
    groups = {'group_1':[1,2,3,4,5,6],'group_2':[7,8,9],'group_3':[10,11,12]} # for dev on exp1 I only have 2015 for now
    # Print fields of source file.
    with h5py.File(file_src,'r') as f :
        for k in [k for k in f.keys()] :
            print('k',k)
            #print(f'{k}: {f[k].dtype}')
    print('start creating the final .h5 file')
    # Create h5.
    with h5py.File(file_src,'r') as fs,h5py.File(file_dest,'w') as fd :
        #for group_name,years in groups.items() :
        for group_name,months in groups.items() :
            logging.info('group_name: %s months: %s',group_name,months)
            grp = fd.create_group(group_name)

            if test2015:
                # Find examples of the specified months.
                indices = np.zeros_like(fs['month'][:],dtype='bool')
                logging.info('month val :%s %s',fs['month'].shape,fs['month'][0].dtype)
                for month in months :
                    indices = np.logical_or(fs['month'][:] == month,indices)
                    print('indices',month,indices.shape,indices.sum())
            else:
                years = []# to remove for dataset on many years
                # Find examples of the specified years.
                indices = np.zeros_like(fs['year'][:],dtype='bool')
                for year in years :
                    indices = np.logical_or(fs['year'][:] == year,indices)
                    print('indices',year,indices.shape,indices.sum())
            # Find examples that don't have nans.
            indices[np.any(np.isnan(fs['py_S'][:]),axis=1)] = 0
            indices[np.isnan(fs['sigma0'][:])] = 0
            indices[np.isnan(fs['normalizedVariance'][:])] = 0
            # Done
            num_examples = indices.sum()
            #print(f'Found {num_examples} events from years: ',years)
            print(f'Found {num_examples} events from months: ',months)

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
            print('timeSAR',fs['timeSAR'].shape)
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
            tmpRe = np.swapaxes(tmpRe,1,2)
            tmpIm = fs['cspcIm'][indices,...].squeeze()
            tmpIm = np.swapaxes(tmpIm,1,2)
            x = np.stack((preprocess.conv_real(tmpRe),
                          preprocess.conv_imaginary(tmpIm),
                          ),
                         axis=3)
            grp.create_dataset('spectrum',data=x)
            print(f'Done with {months}')
            #print(f'Done with {years}')
    print('Done')