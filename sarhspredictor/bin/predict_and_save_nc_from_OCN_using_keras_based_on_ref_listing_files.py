# coding: utf-8
"""
4 janvier 2021
Je sais que la variable S des fichiers de Justin est buggé mais je peut tout de mem lire les fichiers de ces
fichiers de reference et comparer mes predictions avec celles de justin
en utilisant le model quach 2020
"""
import logging
import sys
sys.path.append('/home1/datahome/agrouaze/git/SAR-Wave-Height/')
from load_quach_2020_keras_model import POSSIBLES_MODELS
from predict_with_quach2020_on_OCN_using_keras import main_level_1
import xarray
import pdb
import os
import time
import traceback
def get_fullpath_ocn_from_refdataset(inputfile,dev=False):
    """

    :param inputfile: (str) path
    :param dev: (bool)
    :return:
    """
    logging.debug('input ref file: %s',inputfile)
    ds = xarray.open_dataset(inputfile)
    paths = ds['fileNameL2'].values
    hs_ref = ds['hsNN'].values
    hs_ref_std = ds['hsNNSTD'].values
    print('paths',paths)
    limited_nb_wv = 3000
    limited_nb_wv = 100
    if dev:
        paths = paths[0:limited_nb_wv]
        hs_ref = hs_ref[0:limited_nb_wv]
        hs_ref_std = hs_ref_std[0 :limited_nb_wv]
    logging.info('size: %s',len(paths))
    return paths,hs_ref,hs_ref_std

def predict_and_save(ref_file_input,paths_ocn,outputdir,hs_ref,hs_ref_std,model,modelname):
    newpaths = []
    for ffi,ff in enumerate(paths_ocn):
        ff = ff.decode()
        newpaths.append(ff)
        logging.debug('ff: %s',ff)
    s1_ocn_wv_ds = main_level_1(newpaths,model)
    s1_ocn_wv_ds['hs_ref'] = xarray.DataArray(hs_ref,dims=['time'])
    s1_ocn_wv_ds['hs_ref_std'] = xarray.DataArray(hs_ref_std,dims=['time'])
    logging.debug('s1_ocn_wv_ds : %s',s1_ocn_wv_ds)
    #pdb.set_trace()
    outputpath = os.path.join(outputdir,'Quach2020_ifr_predict_%s_%s'%(modelname,os.path.basename(ref_file_input)))
    s1_ocn_wv_ds.to_netcdf(outputpath)
    logging.info('output : %s',outputpath)
    logging.info('finished')


if __name__ =='__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='validation Quach 2020 against ref dataset')
    parser.add_argument('--verbose',action='store_true',default=False)
    parser.add_argument('--dev',action='store_true',default=False,help='reduce the number of input WV files for dev/test')
    parser.add_argument('--modelversion',action='store',choices=POSSIBLES_MODELS.keys(),required=True,
                        help='possible models: %s' % POSSIBLES_MODELS.keys())
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()
    #onefile_ref = '/home/cercache/users/jstopa/sar/empHs/forAG/S1A_201905S.nc'
    #onefile_ref = '/home/cercache/users/jstopa/sar/empHs/forAG/S1A_201906S.nc'
    #onefile_ref = '/home1/datawork/agrouaze/data/sentinel1/cwave/reference_input_output_dataset_from_Jstopa_quach2020/S1A_201906S.nc'
    try :
        print('first try to  load_quach2020_model')
        model = POSSIBLES_MODELS[args.modelversion]()
    except :
        logging.error('trace : %s',traceback.format_exc())
        print('second try to  load_quach2020_model (except)')#the second try his here because the loading with justin_std have a problem of layer naming solved by two consecutive call
        model = POSSIBLES_MODELS[args.modelversion]()
    onefile_ref = '/home1/datawork/agrouaze/data/sentinel1/cwave/reference_input_output_dataset_from_Jstopa_quach2020/S1A_201901S.nc'
    paths_ocn,hs_ref,hs_ref_std = get_fullpath_ocn_from_refdataset(inputfile=onefile_ref,dev=args.dev)
    outputdir = '/home1/datawork/agrouaze/data/sentinel1/cwave/validation_quach2020/'
    logging.info('outputdir : %s',outputdir)
    predict_and_save(onefile_ref,paths_ocn,outputdir,hs_ref,hs_ref_std,model=model,modelname=args.modelversion)
    logging.info('end of script, elapsed: %1.1f secconds',time.time()-t0)