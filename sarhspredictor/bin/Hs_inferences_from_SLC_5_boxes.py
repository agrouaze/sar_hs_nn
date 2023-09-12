"""
Antoine Grouazel
May 2021
only 5 x spectra and 5 Hs
"""
import logging
import time
import os
import datetime
import sys
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/lib/')
import xsarsea.cross_spectra_core
import numpy as np
import load_quach_2020_keras_model
from prepare_dataset_subdomain_wv import treat_xspectra_fullimage
from sarhspredictor.lib.comp_xspec_on_5_sub_domains import prepare_image_splitting_in_5_domains,comp_xspec_for_one_of_the_5_subimages


def do_hs_inferences_and_save_xspec_and_hs(slc,allspecs,name_subimage):
    """

    :param allspecs:
    :param name_subimage:
    :return:
    """
    logging.info('loading model')
    modelQuach2020 = load_quach_2020_keras_model.load_quach2020_model_v2()
    satellite = os.path.basename(onetiff)[0 :3]

    datedt = datetime.datetime.strptime(os.path.basename(onetiff).split('-')[5],'%Y%m%dt%H%M%S')
    ds_inferences_full_image = treat_xspectra_fullimage(allspecs,satellite,datedt,slc,modelQuach2020,
                                                        use_cwave_lib_without_cartpolconv=False)
    outputfile_full = '/home1/scratch/agrouaze/hs_inferences_slc/5subimages/prediction_hs_over_full_SAR_images_%s_%s_v6.nc' % (
        os.path.basename(onetiff).replace('.tiff',''),name_subimage)
    if os.path.exists(outputfile_full) :
        os.remove(outputfile_full)
    if os.path.exists(os.path.dirname(outputfile_full)) is False :
        os.makedirs(os.path.dirname(outputfile_full))
    ds_inferences_full_image.to_netcdf(outputfile_full)
    logging.info('successfully creation written of output file : %s',outputfile_full)



if __name__ == '__main__' :
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='Hs inferences on 5 sub images SLC')
    parser.add_argument('--verbose',action='store_true',default=False)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    # same image as Noug
    onetiff = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2019/278/S1A_WV_SLC__1SSV_20191005T163939_20191005T171407_029326_035559_2461.SAFE/measurement/s1a-wv1-slc-vv-20191005t165023-20191005t165026-029326-035559-045.tiff'
    # read #############################
    slc_im_da = xsarsea.cross_spectra_core.read_slc(onetiff)
    split_inds_dict = prepare_image_splitting_in_5_domains(slc_im_da)


    for di,onesubim in enumerate(split_inds_dict) :
        sli_az_range = split_inds_dict[onesubim]
        logging.info('%s %s : %s',di,onesubim,sli_az_range)

        allspecs = comp_xspec_for_one_of_the_5_subimages(onetiff,slice_im_az_range=sli_az_range)
        do_hs_inferences_and_save_xspec_and_hs(slc_im_da,allspecs,name_subimage=onesubim)
    logging.info('end of the script')