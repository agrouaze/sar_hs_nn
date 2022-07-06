#!/home/datawork-cersat-public/project/mpc-sentinel1/workspace/conda/envs/satpy/bin/python
"""
A Grouazel
copy paste from https://github.com/hawaii-ai/SAR-Wave-Height/blob/master/scripts/aggregate.ipynb
env : satpy ou xsar_pr46
history:
    06 may 2022: update of the script to run using xarray and produce a netCDF unique file
info run: 1h15min sur noeud datarmor datamem1 avec 1754 files.
    15 juin 2022: avec tte les variables CWAVE + HLF + spec OCN et SLC, ça dure 3,5heures -> 139 Go
"""
# Reads NetCDF4 files and combines them into hdf5 file.
# Author: Peter Sadowski, Dec 2020
import xarray
from netCDF4 import Dataset
import numpy as np
import glob
#import h5py
import sys
import datetime
import traceback
import pandas as pd
try:
    from tqdm import tqdm
    tqdm_load = True
except:
    print('no tqdm')
    tqdm_load = False
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
    if 'exp1' in filename:
        platform,_,_,_,_,_,date,_ext = re.split('[_.]',filename)
        #assert _ext == 'h5',_ext
        logging.debug('date parsed: %s',date)
    else:
        platform,_alt,date,_,_ext = re.split('[_.]',filename)
        assert _alt == 'ALT',_alt
        assert _ext == 'nc',_ext
        assert date[:5] == 'coloc'
        date = date[5 :]
    satellite = int(platform[2] == 'A')  # Encodes type A as 1 and B as 0
    # rval = {'satellite':satellite}

    year = int(date[0 :4])
    month = int(date[4 :6])
    day = int(date[6 :8])
    rval = {'satellite' : satellite,'year' : year,'month' : month,'day':day}
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
    if tqdm_load:
        generator = tqdm(files_src)
    else:
        generator = files_src
    shapes = {}
    for i,filename in enumerate(generator) :
        # Add file of data to large hdf5.
        # print(filename)
        try:
            data = Dataset(filename)
            meta = parse_filename(filename)
            if 'lonSAR' in data.variables.keys() and 'cspcRe_slc' in  data.variables.keys(): #security for files without all the variables
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
                            shapes[key] = maxshape
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
                            shapes[key] = num_prev + num_add
                        for key in meta :
                            num_prev = fdest[key].shape[0]
                            fdest[key].resize(num_prev + num_add,axis=0)
                            fdest[key][-num_add :] = np.ones((data.variables[keys[0]].shape[0],),dtype=int) * meta[key]

            else:
                logging.info('file %s doesnt have lonSAR variable it is probably corrupted',filename)

        except:
            logging.error('impossible to read %s',filename)
            logging.error('traceback :%s', traceback.format_exc())
        logging.info('shape cspcRe_slc : %s shape timeALT %s',shapes['cspcRe_slc'],shapes['timeALT'])

def preproc(ds):
    filee = ds.encoding['source']
    #print('filee',filee)
    #print('sample',len(ds['nsample']))

    meta = parse_filename(filee)
    for vv in  ['satellite','year','month','day']:
        ds[vv] = xarray.DataArray(meta[vv],coords={'nsample':ds['nsample'].values},dims=['nsample'])
    ds = ds.rename({'nsample': 'time'})
    #ds['time'].encoding['units'] = 'seconds since 2014-01-01 00:00:00'
    #print(ds['time'].attrs)
    #del ds['time'].attrs['units']
    #ds['time'].attrs = {'units':'seconds since 2014-01-01 00:00:00'}
    return ds

def xarray_aggregate(input_files,file_dest,keys):
    """

    :param input_files:
    :param file_dest:
    :param keys:
    :return:
    """
    ds = xarray.open_mfdataset(input_files,preprocess=preproc,combine='nested',concat_dim='time')
    logging.info('ds : %s',ds)
    #ds = ds[keys]
     # commented to have all the variables
    logging.info('start to write the destination file')
    glob_attrs = {'agg_step_processing_method': xarray_aggregate.__name__,
                  'agg_step_processing_script': os.path.basename(__file__),
                  'agg_step_processing_env': sys.executable,
                  'agg_step_processing_date': datetime.datetime.today().strftime('%Y%m%d %H:%M'),
                  'agg_step_input_dir': os.path.dirname(input_files[0]),
                  'agg_step_outputdir_dir': os.path.dirname(file_dest)
                  }
    for uu in ds.attrs:
        glob_attrs['normalization_step_%s' % uu] = ds.attrs[uu]
    ds.attrs = glob_attrs
    ds.to_netcdf(file_dest,encoding={'time':{'units':'seconds since 2014-01-01 00:00:00'}})
    logging.info('file_dest written : %s',file_dest)

            #raise mlfkgmk
if __name__ == '__main__':
    import time
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse
    example_inputdir = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v4_norm'
    parser = argparse.ArgumentParser(description='production of figures for cyclic reports CFOSAT SCAT')
    parser.add_argument('--verbose',action='store_true',default=False)
    parser.add_argument('--dev', action='store_true', default=False,help='treat 10 input files just for test/dev')
    parser.add_argument('--inputdir', help='directory where the input nc files are stored for instance %s'%example_inputdir, required=True)
    parser.add_argument('--outputdir',help='directory where the output file will be stored',required=True)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()
    #files_src = sorted(glob.glob("/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/2021/*.nc"))
    #files_src = sorted(glob.glob('/home1/scratch/agrouaze/training_quach_redo_model/*_processed.nc'))
    #files_src = sorted(glob.glob(os.path.join(args.inputdir,'*20180*.nc')))
    files_src = sorted(glob.glob(os.path.join(args.inputdir, '*.nc')))
    if args.dev:
        files_src = files_src[0:10]
    logging.info(f'Found {len(files_src)} files.')
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    # file_dest =  "/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/alt/aggregated_ALT.h5"
    # file_dest =  "/mnt/tmp/psadow/sar/aggregated_ALT.h5"
    # file_dest = "/mnt/tmp/psadow/sar/aggregated_2019.h5"
    # file_dest =  "/mnt/lts/nfs_fs02/sadow_lab/preserve/stopa/sar_hs/data/alt/aggregated_2019.h5"
    #file_dest = os.path.join(args.outputdir,"aggregated.h5")
    #file_dest = os.path.join(args.outputdir, "aggregated_v7_normed.nc")
    file_dest = os.path.join(args.outputdir, "aggregated_v9_normed.nc")

    # keys = ['timeSAR', 'timeALT', 'lonSAR', 'lonALT', 'latSAR', 'latALT', 'hsALT', 'dx', 'dt', 'nk', 'hsSM', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'S']
    # keys = ['timeSAR', 'lonSAR',  'latSAR', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'S']
    # keys += ['cspcRe', 'cspcIm']
    # keys = ['timeSAR', 'lonSAR',  'latSAR', 'incidenceAngle', 'sigma0', 'normalizedVariance', 'py_S', 'cspcRe', 'cspcIm'] #'py_cspcRe', 'py_cspcIm']
    keys = ['timeSAR','timeALT','lonSAR','lonALT','latSAR','latALT','hsALT','wsALT','nk','incidenceAngle','sigma0',
            'normalizedVariance','latlonSARcossin','doySAR','todSAR','incidence','k','phi','spectrum','cspcRe_slc','cspcIm_slc'] # original spectrum are not in normalized daily fiels for exp1v4/v6
    #aggregate(files_src,file_dest,keys=keys)
    xarray_aggregate(files_src, file_dest, keys)
    logging.info('done in %1.2f seconds',(time.time()-t0))