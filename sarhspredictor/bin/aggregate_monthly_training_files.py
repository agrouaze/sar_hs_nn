
"""
A Grouazel
copy paste from https://github.com/hawaii-ai/SAR-Wave-Height/blob/master/scripts/aggregate.ipynb


"""
# Reads NetCDF4 files and combines them into hdf5 file.
# Author: Peter Sadowski, Dec 2020
from netCDF4 import Dataset
import numpy as np
import glob
import h5py
import pandas as pd
from tqdm import tqdm
import re
import os
import logging




def parse_filename ( filename ) :
    """
    Grab some meta data from filename.
    """
    filename = os.path.basename(filename)
    # platform, date, _ext = re.split('_|\.', filename)
    # platform,_alt,date,_ext = re.split('_|\.',filename)
    print(re.split('[_.]',filename))
    platform,_alt,date,_,_ext = re.split('[_.]',filename)
    assert _alt == 'ALT',_alt
    assert _ext == 'nc',_ext
    satellite = int(platform[2] == 'A')  # Encodes type A as 1 and B as 0
    # rval = {'satellite':satellite}
    assert date[:5] == 'coloc'
    date = date[5 :]
    year = int(date[0 :4])
    month = int(date[4 :6])
    rval = {'satellite' : satellite,'year' : year,'month' : month}
    return rval


def process ( x,key ) :
    """
    Process a netcdf variable data.variables[key]
    """
    if key == 'S' :
        x.set_auto_scale(False)
        x = np.array(x[:] * float(x.scale_factor))
    return x


def aggregate ( files_src,file_dest,keys=None ) :
    """
    Aggregate list of netcdf files into single hdf5.
    Args:
    files_src: list of netcdf filenames
    file_dest: filename of h5
    keys: If specified, only extract these fields.
    """

    for i,filename in enumerate(tqdm(files_src)) :
        # Add file of data to large hdf5.
        # print(filename)
        data = Dataset(filename)
        meta = parse_filename(filename)

        if i == 0 :
            if keys is None :
                # Grab keys from first file.
                keys = data.variables.keys()
            with h5py.File(file_dest,'w') as fdest :
                for key in keys :
                    # print(key)
                    x = process(data.variables[key],key)
                    maxshape = (None,) if len(x.shape) == 1 else (None,) + x.shape[1 :]
                    fdest.create_dataset(key,data=x,maxshape=maxshape)
                for key in meta :
                    temp = np.ones((data.variables[keys[0]].shape[0],),dtype=int) * meta[key]
                    fdest.create_dataset(key,data=temp,maxshape=(None,))
        else :
            with h5py.File(file_dest,'a') as fdest :
                for key in keys :
                    num_prev = fdest[key].shape[0]
                    num_add = data.variables[key].shape[0]
                    fdest[key].resize(num_prev + num_add,axis=0)
                    fdest[key][-num_add :] = process(data.variables[key],key)
                for key in meta :
                    num_prev = fdest[key].shape[0]
                    fdest[key].resize(num_prev + num_add,axis=0)
                    fdest[key][-num_add :] = np.ones((data.variables[keys[0]].shape[0],),dtype=int) * meta[key]

if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='production of figures for cyclic reports CFOSAT SCAT')
    parser.add_argument('--verbose',action='store_true',default=False)
    parser.add_argument('--outputdir',help='directory where the figures will be stored',required=True)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')

    #files_src = sorted(glob.glob("/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/2021/*.nc"))
    files_src = sorted(glob.glob('/home1/scratch/agrouaze/training_quach_redo_model/*_processed.nc'))
    print(f'Found {len(files_src)} files.')

    # file_dest =  "/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/alt/aggregated_ALT.h5"
    # file_dest =  "/mnt/tmp/psadow/sar/aggregated_ALT.h5"
    # file_dest = "/mnt/tmp/psadow/sar/aggregated_2019.h5"
    # file_dest =  "/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/alt/aggregated_2019.h5"
    file_dest = os.path.join(args.outputdir,"aggregated.h5")

    # keys = ['timeSAR', 'timeALT', 'lonSAR', 'lonALT', 'latSAR', 'latALT', 'hsALT', 'dx', 'dt', 'nk', 'hsSM', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'S']
    # keys = ['timeSAR', 'lonSAR',  'latSAR', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'S']
    # keys += ['cspcRe', 'cspcIm']
    # keys = ['timeSAR', 'lonSAR',  'latSAR', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'py_S', 'cspcRe', 'cspcIm'] #'py_cspcRe', 'py_cspcIm']
    keys = ['timeSAR','timeALT','lonSAR','lonALT','latSAR','latALT','hsALT','dx','dt','nk','hsSM','incidenceAngle','sigma0',
            'normalizedVariance','cspcRe','cspcIm','py_S']
    aggregate(files_src,file_dest,keys=keys)
    logging.info('done')