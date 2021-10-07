"""
Antoine Grouazel
"""
import time
import xsarsea
import logging
import numpy as np
def comp_xspec_for_one_of_the_5_subimages(onetiff,slice_im_az_range):
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
    # I define the size of the N periodogram equal to the size of the sub image so that N=1 (ie there is no sub domain in the 5 sub domains)
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
    return allspecs

def prepare_image_splitting_in_5_domains(slc,delta_image_limits=4,limited_width_domains=None):
    """

    :param slc:
    :param delta_image_limits: nb pixels to skip on the sides on images to avoid fft artefacts
    :param limited_width_domains: int we could se for instance the rectangle to be 512x512
    :return:
    """
    # order atrack, xtrack
    full_range = slc['sigma0'].squeeze().shape[1]
    half_range = int(np.round(full_range / 2))
    first_quarter = int(np.round(full_range / 4))
    last_quarter = full_range - first_quarter

    full_az = slc['sigma0'].squeeze().shape[0]
    half_az = int(np.round(full_az / 2))
    first_quarter_az = int(np.round(full_az / 4))
    last_quarter_az = full_az - first_quarter_az

    lons_all = slc['longitude'].values
    lats_all = slc['latitude'].values
    if limited_width_domains is None:
        # order atrack xtrack TODO: check whether the position is always top and bottom , Left/right when image is descending
        rect_top_left = (slice(half_az,full_az-delta_image_limits),slice(0+delta_image_limits,half_range,1))
        rect_top_right = (slice(half_az,full_az-delta_image_limits),slice(half_range,full_range-delta_image_limits,1))
        rect_bot_left = (slice(0+delta_image_limits,half_az),slice(0+delta_image_limits,half_range,1))
        rect_bot_right = (slice(0+delta_image_limits,half_az),slice(half_range,full_range-delta_image_limits,1))
        rec_crop_center = (slice(first_quarter_az,last_quarter_az),slice(first_quarter,last_quarter))
    else:
        assert half_az+limited_width_domains<=full_az
        assert half_range + limited_width_domains <= full_range
        rect_top_left = (slice(half_az+first_quarter_az,half_az+first_quarter_az+limited_width_domains),
                         slice(first_quarter ,first_quarter+limited_width_domains,1))
        rect_top_right = (slice(half_az+first_quarter_az,half_az+first_quarter_az+limited_width_domains),
                          slice(half_range+first_quarter,half_range+first_quarter+limited_width_domains,1))
        rect_bot_left = (slice(first_quarter_az,first_quarter_az+limited_width_domains),
                         slice(first_quarter ,first_quarter+limited_width_domains,1))
        rect_bot_right = (slice(first_quarter_az,first_quarter_az+limited_width_domains),
                          slice(half_range+first_quarter,half_range+first_quarter+limited_width_domains,1))
        rec_crop_center = (slice(half_az-int(limited_width_domains/2),half_az+int(limited_width_domains/2)),
                           slice(half_range-int(limited_width_domains/2),half_range+int(limited_width_domains/2)))
    # names of images are given in the convention : bottom is begining of image in azimuth and left is begining of range
    didi = {'rect_top_left' : rect_top_left,
            'rect_top_right' : rect_top_right,
            'rect_bot_left' : rect_bot_left,
            'rect_bot_right' : rect_bot_right,
            'rec_crop_center' : rec_crop_center}
    # store long/lat of each rectangles
    geoloc = {}
    for rect_x in didi:
        sli_az,sli_ra = didi[rect_x]
        lons = [
            lons_all[sli_az.start,sli_ra.start],
            lons_all[sli_az.start,sli_ra.stop],
            lons_all[sli_az.stop,sli_ra.stop],
            lons_all[sli_az.stop,sli_ra.start],
            lons_all[sli_az.start,sli_ra.start],
        ]
        lats = [
            lats_all[sli_az.start,sli_ra.start],
            lats_all[sli_az.start,sli_ra.stop],
            lats_all[sli_az.stop,sli_ra.stop],
            lats_all[sli_az.stop,sli_ra.start],
            lats_all[sli_az.start,sli_ra.start],
        ]
        geoloc[rect_x] = {'lons':np.array(lons),'lats':np.array(lats)}
    return didi,geoloc