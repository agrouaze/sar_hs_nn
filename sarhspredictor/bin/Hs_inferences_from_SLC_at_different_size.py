# coding: utf-8
"""
I want to perform a computation of Hs with different size of sub image
May 2021
developed with env : xsarQuach2020
v3 on GPU machines with wrong shape X spectra cwave full of NaN but strangly Hs are ok...
v4 : on CPU machines with ok shape 60,70 as input of CWAVE computation (still with normalization to have a max at 60 for re part), merde qlq v4 sont a refaire pcq fait avec normalization v5
v5 : same as v4 but normalization factor is 45000
"""
import logging
import time
import os
import datetime
import sys
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/lib/')
import xsarsea.cross_spectra_core
#same image as Noug
onetiff = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2019/278/S1A_WV_SLC__1SSV_20191005T163939_20191005T171407_029326_035559_2461.SAFE/measurement/s1a-wv1-slc-vv-20191005t165023-20191005t165026-029326-035559-045.tiff'
import xsar
# definie a listing
import numpy as np
import load_quach_2020_keras_model
from prepare_dataset_subdomain_wv import treat_xspectra_fullimage

def inference_for_one_image_subsize(delta):
    t0 = time.time()
    sli = {'range' : slice(center_range - delta,center_range + delta,1),
           'azimuth' : slice(center_azimuth - delta,center_azimuth + delta,1)}

    all_slices.append(sli)
    # read #############################
    slc = xsarsea.cross_spectra_core.read_slc(onetiff,slice_subdomain=sli)
    # compute X spectra ###############################
    subimage_range = delta * 2
    subimage_azimuth = delta * 2
    # subimage_range=1024
    # subimage_azimuth=1024
    # subimage_range = 2048
    # subimage_azimuth = 2048
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
    outputfile_full = '/home1/scratch/agrouaze/hs_inferences_slc/image_size_test/prediction_hs_over_full_SAR_images_%s_%sx%s_v5.nc' % (
        os.path.basename(onetiff).replace('.tiff',''),subimage_azimuth,subimage_range)
    if os.path.exists(outputfile_full) :
        os.remove(outputfile_full)
    if os.path.exists(os.path.dirname(outputfile_full)) is False :
        os.makedirs(os.path.dirname(outputfile_full))
    ds_inferences_full_image.to_netcdf(outputfile_full)
    logging.info('successfully creation written of output file : %s',outputfile_full)

    # dsinfFullIm = do_inferences_from_slc_xspectra(slc,datedt,satellite,
    #                                                model=model,kx_ori=kx_ori
    #                                                ,ky_ori=ky_ori,
    #                                                crossSpectraRe=crossSpectraRe,
    #                                                crossSpectraIm=crossSpectraIm,
    #                                                use_cwave_lib_without_cartpolconv=use_cwave_lib_without_cartpolconv)






if __name__ =='__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='Hs inferences at different size')
    parser.add_argument('--verbose',action='store_true',default=False)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    center_range = 2200  # au doigt leve
    center_azimuth = 2300
    all_slices = []
    possible_semi_span = np.arange(20,2000,30)
    for di,delta in enumerate(possible_semi_span) :
        logging.info('delta :%s  %s/%s',delta * 2,di,len(possible_semi_span))
        inference_for_one_image_subsize(delta)
    logging.info('end of the script')



