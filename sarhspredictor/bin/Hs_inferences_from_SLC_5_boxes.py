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

def inference_for_one_of_the_5_subimages(onetiff,name_subimage,slice_im_az_range):
    """

    :param name_subimage:
    :param slice_im_az_range:
    :return:
    """
    t0 = time.time()
    # read #############################
    slice_im = {}
    slice_im['azimuth'] = slice_im_az_range[0]
    slice_im['range'] = slice_im_az_range[1]
    slc = xsarsea.cross_spectra_core.read_slc(onetiff,slice_subdomain=slice_im)
    # compute X spectra ###############################
    # I define the size of the N periodogram equal to the size of the sub image so that N=1
    subimage_range = slice_im['range'].stop - slice_im['range'].start
    subimage_azimuth = slice_im['azimuth'].stop - slice_im['azimuth'].start
    overlap_size = 0
    # overlap_size = 0
    allspecs,frange,fazimuth,allspecs_per_sub_domain,splitting_image,limits_sub_images = \
        xsarsea.cross_spectra_core.compute_SAR_cross_spectrum(slc,
                                                              N_look=3,look_width=0.25,
                                                              look_overlap=0.,look_window=None,range_spacing=None,
                                                              welsh_window='hanning',
                                                              nperseg={'range' : subimage_range,
                                                                       'azimuth' : subimage_azimuth},
                                                              noverlap={'range' : overlap_size,'azimuth' : overlap_size}
                                                              ,spacing_tol=1e-3)
    logging.info('time to get %s X-spectra : %1.1f seconds',len(allspecs_per_sub_domain),time.time() - t0)
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
    slc = xsarsea.cross_spectra_core.read_slc(onetiff)
    # order atrack, xtrack
    full_range = slc['sigma0'].values.squeeze().shape[1]
    half_range = int(np.round(full_range / 2))
    first_quarter = int(np.round(full_range / 4))
    last_quarter = full_range - first_quarter

    full_az = slc['sigma0'].values.squeeze().shape[0]
    half_az = int(np.round(full_az / 2))
    first_quarter_az = int(np.round(full_az / 4))
    last_quarter_az = full_az - first_quarter_az

    # order atrack xtrack
    rect_top_left = (slice(half_az,full_az),slice(0,half_range,1))
    rect_top_right = (slice(half_az,full_az),slice(half_range,full_range,1))
    rect_bot_left = (slice(0,half_az),slice(0,half_range,1))
    rect_bot_right = (slice(0,half_az),slice(half_range,full_range,1))
    rec_crop_center = (slice(first_quarter_az,last_quarter_az),slice(first_quarter,last_quarter))

    didi = {'rect_top_left' : rect_top_left,
            'rect_top_right' : rect_top_right,
            'rect_bot_left' : rect_bot_left,
            'rect_bot_right' : rect_bot_right,
            'rec_crop_center' : rec_crop_center}


    for di,onesubim in enumerate(didi) :
        sli_az_range = didi[onesubim]
        logging.info('%s %s : %s',di,onesubim,sli_az_range)

        inference_for_one_of_the_5_subimages(onetiff,name_subimage=onesubim,slice_im_az_range=sli_az_range)
    logging.info('end of the script')