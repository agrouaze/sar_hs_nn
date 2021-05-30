"""
May 2021
A grouazel
I will do a nw training dataset replacing the X polar cross spectra from L2 ESA by X spectra computed from SLC
This operation will be quite long for each measurements so I want to extract each dates present in the cwaveV4 dataset
{'L1_not_found': 51882, 'L1_found': 714544}: 7.2% de missing
"""
import time
import os
import sys
import numpy as np
import logging
from collections import defaultdict
import datetime
import glob
import xarray
sys.path.append('/home1/datahome/agrouaze/git/mpc/data_collect')
import match_L1_L2_measurement #data_collect (mpc repo)
DIR_INPUT = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/quach2020/validation/colocations/original_colocations_YoungAltiDatabase_vs_WV_L2_Jstopa/cwaveV4/'

if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='extract dates')
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

    list_monthly_nc = sorted(glob.glob(os.path.join(DIR_INPUT,'*.nc')))
    logging.info('Nb files: %s',len(list_monthly_nc))
    all_fnames = []
    all_fnames_tiff_slc = []
    cpt = defaultdict(int)
    #ds = xarray.open_mfdataset(list_monthly_nc,combine='nested',concat_dim='time')
    for ffi,ff in enumerate(list_monthly_nc):
        logging.info('ff %s/%s cpt: %s',ffi,len(list_monthly_nc),cpt)
        ds = xarray.open_dataset(ff)
        #logging.info('ds : %s',ds)
        #all_dates = []
        #logging.info('vals: %s',ds['fileNameFull'].values)
        for iiu in range(len(ds['fileNameFull'].values)):
            tmp_values = []
            #for iiu in range(tmpva_iin[:].shape[0]) :
            #apath = ('').join([ddc.decode() for ddc in ds['fileNameFull'].values[iiu]])
            #tmp_values.append(apath.replace('cercache','datawork-cersat-public'))
            single = ds['fileNameFull'].values[iiu].decode().replace('cercache','datawork-cersat-public')
            single_slc = match_L1_L2_measurement.getTiffcorresponding2NetCDF(single)
            all_fnames.append(single)
            if single_slc != '':
                cpt['L1_found'] += 1
                all_fnames_tiff_slc.append(single_slc)
            else:
                cpt['L1_not_found'] += 1
                all_fnames_tiff_slc.append('nan')

            #input_values = np.array(tmp_values)
            #logging.info('single : %s',single)
            #sdmsksmdlk
    logging.info('done %s',cpt)
    output_listing = os.path.join(args.outputdir,'listing_SAR_L2_L1_measu_from_colocations_cwaveV4.txt')
    fid = open(output_listing,'w')
    for yyi,yy in enumerate(all_fnames_tiff_slc):
        fid.write(all_fnames_tiff_slc[yyi]+' '+all_fnames[yyi]+'\n')
    fid.close()
    logging.info('output listing : %s',output_listing)