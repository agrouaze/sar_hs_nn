"""
June 2022
Grouazel
conda: xsar_pr46:
sur datamem1: 2500 daily files -> 1h30min
3500 files -> 2.2 heures en netCDF
sur noeud mpi en .zarr: 379k files en 2 heures.
4094 files en zarr -> 2.6h
4282 files (503017 pts) -> 3.6h
"""
import os
from dask.distributed import Client, LocalCluster
import logging
import xarray
import datetime
import sys
import glob
import numpy as np

def preproc_analogs(ds):
    filee = ds.encoding['source']
    keys_to_drop = ['dk', 'crossSpectraReCart_tau2', 'crossSpectraImCart_tau2', 'crossSpectraImPol', #'CWAVE_20_SLC',
                    'tiff']
    if np.any(ds['phi']>4):
        ds['phi'] = xarray.DataArray(np.radians(np.arange(0,360,5)))
    for vv in keys_to_drop:
        ds = ds.drop(vv)
    return ds

def preproc_Hs_NN_DL(ds):
    filee = ds.encoding['source']
    #keys_to_drop = ['dk', 'crossSpectraReCart_tau2', 'crossSpectraImCart_tau2']
    if np.any(ds['phi']>4):
        ds['phi'] = xarray.DataArray(np.radians(np.arange(0,360,5)))
    #for vv in keys_to_drop:
    #    ds = ds.drop(vv)
    return ds

def xarray_aggregate(input_files,file_dest,application='analogs'):
    """

    :param input_files:
    :param file_dest:
    :return:
    """
    if application == 'analogs':
        ds = xarray.open_mfdataset(input_files,preprocess=preproc_analogs,combine='nested',concat_dim='time_sar')
    elif application == 'hsDL':
        ds = xarray.open_mfdataset(input_files, preprocess=preproc_Hs_NN_DL, combine='nested', concat_dim='nsample')
    else:
        raise Exception('unknown application')
    logging.info('ds : %s',ds)
    #ds = ds[keys]
     # commented to have all the variables

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
    return ds

def write_to_disk(ds,file_dest):
    """

    :param ds:
    :param file_dest:
    :param out:
    :return:
    """

    if 'nc' in file_dest:
        if 'time_sar' in ds:
            ds.to_netcdf(file_dest,encoding={'time_sar':{'units':'seconds since 2014-01-01 00:00:00'}})
        else:
            ds.to_netcdf(file_dest)
    elif 'zarr' in file_dest:
        ds.to_zarr(file_dest)
    else:
        raise Exception('format of outpufile unkown : %s',os.path.basename(file_dest))
    logging.info('file_dest written : %s',file_dest)
    return 1

            #raise mlfkgmk
if __name__ == '__main__':
    import time
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse
    apps = ['analogs','hsDL']
    example_inputdir = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp2D4/v1/'
    parser = argparse.ArgumentParser(description='aggregattion of training files for analogs and Deep learning')
    parser.add_argument('--verbose',action='store_true',default=False)
    parser.add_argument('--dev', action='store_true', default=False,help='treat 10 input files just for test/dev')
    parser.add_argument('--inputdir', help='directory where the input nc files are stored for instance %s'%example_inputdir, required=True)
    parser.add_argument('--outputdir',help='directory where the output file will be stored',required=True)
    parser.add_argument('--application',choices=apps,help='which application the future file is designed to %s '%apps)
    parser.add_argument('--format',choices=['nc','zarr'],help='zarr or nc',required=False,default='zarr')
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()
    if args.application == 'analogs':
        files_src = sorted(glob.glob(os.path.join(args.inputdir,'*','*', '*.nc')))
    elif args.application == 'hsDL':
        files_src = sorted(glob.glob(os.path.join(args.inputdir, '*', '*.nc')))
    else:
        raise Exception('not handle application')
    logging.info(f'Found {len(files_src)} files.')
    if args.dev:
        logging.info('nb files reduction for dev')
        files_src = files_src[0:15]
    logging.info(f'Found {len(files_src)} files.')
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    if args.format=='zarr':
        exte = '.zarr'
    elif args.format=='nc':
        exte = '.nc'
    else:
        raise Exception('format not handle')
    file_dest = os.path.join(args.outputdir, "aggregated_exp2_v1.5"+exte)
    file_dest = os.path.join(args.outputdir, "aggregated_exp2_v2.0_"+ args.application +'_'+ exte) #5 June:  colocs 2019-2022 with azimuth cutoff
    ds = xarray_aggregate(files_src, file_dest,application=args.application)
    if args.application == 'analogs':
        ds = ds.chunk({'phi':72,'k':60,'cwave_coords':20,'kx':164,'ky':84,'time_sar':20000})
    elif args.application == 'hsDL':
        ds = ds.chunk({'phi': 72, 'k': 60, 'cwave_coord': 20, 'nsample': 20000,'latSARcossinlonSARcossin':4})
    logging.info('start to write the destination file : %s',file_dest)
    if os.path.exists(file_dest):
        logging.warning('destination file already exists!!')
    # if False:
    #     cluster = LocalCluster(processes=True) #n_workers=2, threads_per_worker=2, processes=True
    #     logging.info('cluster: %s',cluster)
    #     client = Client(cluster)
    #     toto = client.submit(write_to_disk,ds,file_dest)
    #     #write_to_disk(ds, file_dest, out=args.format)
    # else:
    write_to_disk(ds,file_dest)
    logging.info('done in %1.2f seconds',(time.time()-t0))