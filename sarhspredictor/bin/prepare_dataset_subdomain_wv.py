# coding: utf-8
"""
May 2021
Grouazel
https://trello.com/c/KbTJOmpj/538-trouver-des-hh-vv-avec-m%C3%AAme-hs-plutot-en-wv2-et-m%C3%AAme-%C3%A9tat-de-mer
contenu du dataset:
ce dataset va servir a faire des inferences (pas a faire un nouveau model )
<NRCS> per sub domain
<Nv>
<cwave>

choix du dataset WV SLC d'origine,
1er dataset: ne seule image (celle de Nouguier) ou  bien une image avec homogène
environement: xsarQuach2020
"""

import sys
import time
import os
import datetime
import numpy as np
import logging
import xarray
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
#sys.path.append('/home1/datahome/agrouaze/sources/wave_inversion')
#sys.path.append('/home1/datahome/agrouaze/git/offline_xrft/')
#from compute_cross_spectra_using_xsar_and_nouguier_core import read_slc
#import cross_spectra_core_NouguierApril2021
import xsarsea
import xsarsea.cross_spectra_core
import xsarsea.spectrum_clockwise_to_trigo
import xsarsea.spectrum_rotation
import xsarsea.conversion_polar_cartesian
sys.path.append('/home1/datahome/agrouaze/git/mpc/data_collect')
from reference_oswk import reference_oswK_1145m_60pts
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/lib/')
from compute_CWAVE_params_from_cart_Xspectra import format_input_CWAVE_vector_from_SLC
from compute_CWAVE_params import format_input_CWAVE_vector_from_OCN
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/lib/sarhs')
import preprocess
import predict_with_quach2020_on_OCN_using_keras
import load_quach_2020_keras_model
#from conversion_polar_cartesian_pyabel import reproject_image_into_polar
#import conversion_polar_cartesian_nouguier_v2


from get_all_headings import get_heading_from_corners
from get_L2_imagette_corners import order_the_corners


def do_inferences_from_slc_xspectra(subset,datedt,satellite,model,kx_ori,ky_ori,crossSpectraRe,crossSpectraIm,
                                    use_cwave_lib_without_cartpolconv=False):
    """

    :param subset: DatasetXarray piece of the full image (could also be the full image itself (in such situation subset==slc) (xsar obj)
    :param datedt: datetime
    :param satellite: str
    :param model: obj keras model to predict the Hs (Quach 2020 algo)
    :param kx_ori: cartesian coordinates in X for the Xspectra computed on the subdomain vect 1d of the subset
    :param ky_ori: cartesian coordinates in Y for the Xspectra computed on the subdomain  vect 1d
    :param crossSpectraRe: Dataarray realpart of the cross spectra of sub image
    :param crossSpectraIm:Dataarray imaginary part of the cross spectra of sub image
    :param use_cwave_lib_without_cartpolconv: bool
    :return:
    """
    t_gather_param0 = time.time()

    mat_s0_dB = 10. * np.log10(subset['sigma0'].values)
    mask_finite_s0 = np.isfinite(mat_s0_dB)
    mean_s0 = np.mean(mat_s0_dB[mask_finite_s0])
    logging.info('mean_s0 : %s %s elapsed :%s seconds',type(mean_s0),mean_s0,time.time() - t_gather_param0)
    # mean_s0 = np.array([-5.46]) #for test /dev je force avec la valeur moyenne indiquer dans le L2 -> ca ne change pas le problem de prediction
    # code from LOP lop_osw_application.py L 3113
    image = subset['digital_number'].values
    dummy = (image * np.conj(image)).real
    intensity = np.mean(dummy)
    normvar = np.var(dummy,ddof=1) / intensity ** 2.
    logging.info('normvar %s %s elapsed :%s seconds',type(normvar),normvar,time.time()-t_gather_param0)
    t_geoloc = time.time()
    central_xrange_ind = int(subset['incidence'].shape[1] / 2)
    central_azimuth_ind = int(subset['incidence'].shape[0] / 2)
    central_inc = subset['incidence'].values[central_xrange_ind,central_azimuth_ind]
    central_lon = subset['longitude'].values[central_xrange_ind,central_azimuth_ind]
    central_lat = subset['latitude'].values[central_xrange_ind,central_azimuth_ind]
    logging.info('central_inc %s central_lon %s central_lat %s (elapsed %s seconds)',central_inc,central_lon,central_lat,time.time()-t_geoloc)
    # step 4 convert cross spectra of each sub images into k,phi polar grid
    # skip this step since there is a pol->cartesian conversion in the method to get CWAVE params I can avoid
    # step 5 compute C-wave 22 params on cross spectra

    # step 6 convert from cartesian grid to polar grid
    logging.debug('kx_ori : %s',kx_ori.shape)
    logging.debug('crossSpectraRe : %s',crossSpectraRe.shape)
    # crossSpectraRePol,rDrid2D,thetaGrid2D = reproject_image_into_polar(crossSpectraRe.squeeze(),rowi=ky_ori,coli=kx_ori,
    #                                                                    theta_i=np.degrees(np.arange(0,360,5)),
    #                                                                    r_i=reference_oswK_1145m_60pts,Jacobian=False)
    # new_spec_PolarRe = conversion_polar_cartesian.from_xCartesianSpectrum(crossSpectraRe,Nphi=72,
    #                                                                                 ksampling='log',**{'Nk' : 60})
    t_polcaart = time.time()
    new_spec_PolarRe = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(crossSpectraRe,Nphi=72,
                                                                                  ksampling='log',**{'Nk' : 60,'kmin' :
            reference_oswK_1145m_60pts[0],'kmax' : reference_oswK_1145m_60pts[-1]})
    logging.debug('new_spec_Polar : %s',new_spec_PolarRe)
    logging.info('new_spec_PolarRe. elapsed %s s',time.time()-t_polcaart)

    new_spec_PolarIm = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(crossSpectraIm,Nphi=72,
                                                                                  ksampling='log',**{'Nk' : 60,'kmin' :
            reference_oswK_1145m_60pts[0],'kmax' : reference_oswK_1145m_60pts[-1]})
    logging.debug('new_spec_PolarIm : %s',new_spec_PolarIm)

    crossSpectraRePol = new_spec_PolarRe  # .values.squeeze()

    crossSpectraRePol = xsarsea.spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(crossSpectraRePol)
    crossSpectraRePol = xsarsea.spectrum_rotation.apply_rotation(crossSpectraRePol,
                                                                 90.)  # This is for having origin at North
    crossSpectraRePol = xsarsea.spectrum_rotation.apply_rotation(crossSpectraRePol,subset.attrs['platform_heading'])

    crossSpectraImPol = new_spec_PolarIm  # .values.squeeze()
    crossSpectraImPol = xsarsea.spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(crossSpectraImPol)
    crossSpectraImPol = xsarsea.spectrum_rotation.apply_rotation(crossSpectraImPol,
                                                                 90.)  # This is for having origin at North
    crossSpectraImPol = xsarsea.spectrum_rotation.apply_rotation(crossSpectraImPol,subset.attrs['platform_heading'])
    # change amplitude of X spectra to match expected range in ESA L2 (patch ugly)
    logging.debug('change amplitude X spectra (current max : %1.5f)',np.amax(crossSpectraRePol))
    divisor = np.amax(crossSpectraRePol.values)/60. #
    divisor = 45000.
    #crossSpectraRePol = crossSpectraRePol / 45000.
    #crossSpectraImPol = crossSpectraImPol / 45000.
    logging.info('artifical normalization factor for polar cross spectra: %s',divisor)
    crossSpectraRePol = crossSpectraRePol/divisor
    crossSpectraImPol = crossSpectraImPol / divisor
    logging.debug('crossSpectraRePol = %s new max : %1.5f',crossSpectraRePol.shape,np.amax(crossSpectraRePol))
    crossSpectraRePol = crossSpectraRePol.squeeze()
    crossSpectraImPol = crossSpectraImPol.squeeze()

    logging.debug('crossSpectraRePol : %s',crossSpectraRePol.shape)

    pt1 = (subset['longitude'].values[0,0],subset['latitude'].values[0,0])
    pt2 = (subset['longitude'].values[0,-1],subset['latitude'].values[0,-1])
    pt3 = (subset['longitude'].values[-1,-1],subset['latitude'].values[-1,-1])
    pt4 = (subset['longitude'].values[-1,0],subset['latitude'].values[-1,0])
    logging.debug('pt4 : %s',pt4)
    corners_L1 = [pt1,pt2,pt3,pt4]
    corners_L1 = order_the_corners(corners_L1)
    lonsL1 = np.array([corners_L1[uu][0] for uu in range(len(corners_L1))])
    latsL1 = np.array([corners_L1[uu][1] for uu in range(len(corners_L1))])
    lonsL1 = np.append(lonsL1,lonsL1[0])
    latsL1 = np.append(latsL1,latsL1[0])
    logging.debug('lonsL1: %s',lonsL1)
    logging.debug('latsL1: %s',latsL1)
    # image_heading,A,B = get_heading_from_corners(lonsL1,latsL1,oswHeading=subset.attrs['platform_heading'],

    #                                              corners=corners_L1,edge='right')
    # logging.debug('image_heading = %1.3f platform pointing : %s (ie %1.3f)',image_heading,subset.attrs['platform_heading'],
    #               subset.attrs['platform_heading'] % 360)
    image_heading = subset.attrs['platform_heading'] #small trick to skip the image heading computation which is almost the same as platform heading in my test image
    logging.info('time to gather params to do the inference (except cwave) :%1.1f seconds',time.time()-t_gather_param0)
    if use_cwave_lib_without_cartpolconv :
        # this case is not used
        subset_ok,_,_,_,_,_,S = format_input_CWAVE_vector_from_SLC(cspcRe=crossSpectraRe,cspcIm=crossSpectraIm,
                                                                   incidenceangle=central_inc,
                                                                   s0=mean_s0,nv=normvar,kx_ori=kx_ori,
                                                                   ky_ori=ky_ori,datedt=datedt,lonSAR=central_lon,
                                                                   latSAR=central_lat,
                                                                   satellite=satellite)
    else :

        logging.debug('cartesian X spectra from which I want to extract cwave params : %s',crossSpectraRePol.shape)
        logging.debug('any NaN in the X spectra pol ? %s %s',np.isnan(crossSpectraRePol).any(),
                      np.isnan(crossSpectraImPol).any())
        t_cwave0 = time.time()
        assert crossSpectraRePol.shape == (60,72)  # case only one spectra
        subset_ok,_,_,_,_,ks1,ths1,_,_,_,S = format_input_CWAVE_vector_from_OCN(crossSpectraRePol.values,crossSpectraImPol.values,
                                                                                ths1=np.arange(0,360,5),
                                                                                ta=image_heading,
                                                                                incidenceangle=central_inc,
                                                                                s0=mean_s0,nv=normvar,
                                                                                ks1=reference_oswK_1145m_60pts,
                                                                                datedt=datedt,lonSAR=central_lon,
                                                                                latSAR=central_lat,
                                                                                satellite=satellite)
        logging.debug('time to compute cwave params : %1.3f',time.time()-t_cwave0)
        assert np.isfinite(S).any()
    # step 6 : normalize the params using sarhspredictor lib
    cwave = np.hstack([S.T,mean_s0.reshape(-1,1),normvar.reshape(-1,1)])  # found L77 in preprocess.py
    cwave = preprocess.conv_cwave(cwave)
    dx = preprocess.conv_dx(np.array([0]))  # delta colocs with alti ... I dont care for inference
    dt = preprocess.conv_dt(np.array([1]))
    latSARcossin = preprocess.conv_position(central_lat)  # Gets cos and sin
    lonSARcossin = preprocess.conv_position(central_lon)
    latlonSARcossin = np.hstack([latSARcossin,lonSARcossin])
    crossSpectraRePol = crossSpectraRePol.T # from 60,72 -> 72,60
    crossSpectraImPol = crossSpectraImPol.T
    crossSpectraRe_conv = preprocess.conv_real(crossSpectraRePol.values.reshape((1,72,60)))
    crossSpectraIm_conv = preprocess.conv_imaginary(crossSpectraImPol.values.reshape((1,72,60)))
    spectrum = np.stack((crossSpectraRe_conv,crossSpectraIm_conv),axis=3)
    # tod = preprocess.conv_time(datedt)
    tod = subset_ok['todSAR']
    incidence = preprocess.conv_incidence(np.array([central_inc]))

    ds = xarray.Dataset()
    dimszi = ['time','cwavedim']
    coordi = {'time' : [datedt],'cwavedim' : np.arange(22)}
    ds['cwave'] = xarray.DataArray(data=cwave,dims=dimszi,coords=coordi)

    dimszi = ['time','Sdim']
    coordi = {'time' : [datedt],'Sdim' : np.arange(20)}
    ds['S'] = xarray.DataArray(data=S.T,dims=dimszi,coords=coordi)

    dxdt = np.column_stack([dx,dt])
    dimszi = ['time','dxdtdim']
    coordi = {'time' : [datedt],'dxdtdim' : np.arange(2)}
    ds['dxdt'] = xarray.DataArray(data=dxdt,dims=dimszi,coords=coordi)

    dimszi = ['time','latlondim']
    coordi = {'time' : [datedt],'latlondim' : np.arange(4)}
    ds['latlonSARcossin'] = xarray.DataArray(data=latlonSARcossin,dims=dimszi,coords=coordi)

    dimszi = ['time']
    coordi = {'time' : [datedt]}
    ds['todSAR'] = xarray.DataArray(data=tod,dims=dimszi,coords=coordi)

    dimszi = ['time','incdim']
    coordi = {'time' : [datedt],'incdim' : np.arange(2)}
    ds['incidence'] = xarray.DataArray(data=incidence,dims=dimszi,coords=coordi)

    dimszi = ['time']
    coordi = {'time' : [datedt]}
    satellite_int = np.array([satellite[2] == 'a']).astype(int)
    ds['satellite'] = xarray.DataArray(data=satellite_int,dims=dimszi,coords=coordi)

    # add variables not needed for inferences but for interpretations
    dimszi = ['time',]
    coordi = {'time' : [datedt]}
    ds['sigma0'] = xarray.DataArray(data=mean_s0,dims=dimszi,coords=coordi)

    dimszi = ['time',]
    coordi = {'time' : [datedt]}
    ds['nv'] = xarray.DataArray(data=normvar,dims=dimszi,coords=coordi)

    dimszi = ['time',]
    coordi = {'time' : [datedt]}
    ds['lon'] = xarray.DataArray(data=central_lon,dims=dimszi,coords=coordi)

    dimszi = ['time',]
    coordi = {'time' : [datedt]}
    ds['lat'] = xarray.DataArray(data=central_lat,dims=dimszi,coords=coordi)

    # I add the X spectra Re and Im to help debuging/analysing outputs/inputs inferences
    coordi = {}
    coordi['time'] = [datedt]
    coordi['k'] = crossSpectraRePol.k.values
    coordi['phi'] = crossSpectraRePol.phi.values
    dimsadd = ['time','phi','k']
    logging.debug('crossSpectraRePol before saving : %s',crossSpectraRePol.shape)
    if crossSpectraRePol.shape == (72,60) :  # case only one spectra
        crossSpectraRePol = crossSpectraRePol.values.reshape((1,72,60))
    ds['xspecRePol'] = xarray.DataArray(data=crossSpectraRePol,dims=dimsadd,coords=coordi)
    if crossSpectraImPol.shape == (72,60) :  # case only one spectra
        crossSpectraImPol = crossSpectraImPol.values.reshape((1,72,60))
    #
    ds['xspecImPol'] = xarray.DataArray(data=crossSpectraImPol,dims=dimsadd,coords=coordi)
    # I also add the sub piece of images DN
    ds['digitalnumber_SubImage_real'] = xarray.DataArray(subset['digital_number'].values.real,
                                                         dims=subset['digital_number'].dims,
                                                         coords=subset['digital_number'].coords)
    ds['digitalnumber_SubImage_imag'] = xarray.DataArray(subset['digital_number'].values.imag,
                                                         dims=subset['digital_number'].dims,
                                                         coords=subset['digital_number'].coords)

    # step 7 : define final features and predict
    t_predict0 = time.time()
    features = predict_with_quach2020_on_OCN_using_keras.define_features(ds)
    input_values = predict_with_quach2020_on_OCN_using_keras.define_input_test_dataset(features,spectrum)
    yhat = predict_with_quach2020_on_OCN_using_keras.do_my_prediction(model,input_values)
    logging.debug('time to do the prediction: %1.3f seconds',time.time()-t_predict0)
    hs_predicted = yhat[:,0]
    hs_uncertainty = yhat[:,1]
    ds['HsQuach'] = xarray.DataArray(data=hs_predicted,dims=['time'])
    ds['HsQuach_uncertainty'] = xarray.DataArray(data=hs_uncertainty,dims=['time'])
    logging.debug('hs Quach : %1.4f m',hs_predicted)
    return ds

def treat_xspectra_one_subimage(allspecs_per_sub_domain,subdomain,satellite,datedt,slc,splitting_image,model,
                                use_cwave_lib_without_cartpolconv=False):
    """
     wrapper before Hs inferences for sub images
    :param allspecs_per_sub_domain: dict containing the xarray dataset with cross spectra and different look 0 tau , 1 tau ,...
    :param subdomain: (int) indice giving the subdomain to process
    :param satellite: (str) 's1a' or ...
    :param datedt: (datetime.datetime) date of the SAR image (starting date in filename is enough) for this application
    :param slc: (xarray dataset) containing DN sigma0 and other variable expected by xsar reader and S-1 SLC data
    :param splitting_image: (dict) containing the slice in range and azimuth of the sub images
    :param model: (keras model loaded)
    :param use_cwave_lib_without_cartpolconv (bool): I tuned the official method to get cwave params to be able to process directly cartesian cross spectra
    :return:
    """
    crossSpectraRe = np.abs(allspecs_per_sub_domain[subdomain]['cross-spectrum_2tau'].mean(dim='2tau').real) #why is there a absolute value here???
    crossSpectraIm = allspecs_per_sub_domain[subdomain]['cross-spectrum_2tau'].mean(dim='2tau').imag
    # cspRe_okGrid = crossSpectraRe.assign_coords(kx=crossSpectraRe.kx.data / (1.1 * np.pi),
    #                                  ky=crossSpectraRe.ky.data / (1.1 * np.pi))  # to to match the scale of Nouguier
    # cspIm_okGrid = crossSpectraIm.assign_coords(kx=crossSpectraIm.kx.data / (1.1 * np.pi),
    #                                  ky=crossSpectraIm.ky.data / (1.1 * np.pi))  # to to match the scale of Nouguier
    kx_ori = allspecs_per_sub_domain[subdomain]['kx'].values
    ky_ori = allspecs_per_sub_domain[subdomain]['ky'].values
    rect = splitting_image[subdomain]
    t_isel = time.time()
    subset = slc.isel(azimuth=rect['azimuth'],range=rect['range'])
    logging.info('time to isel in the image: %s',time.time()-t_isel)
    t_pers = time.time()
    subset['sigma0'] = subset['sigma0'].persist() # test to see if it increase the speed to compute the hs
    subset['incidence'] = subset['incidence'].persist()
    subset['longitude'] = subset['longitude'].persist()
    subset['latitude'] = subset['latitude'].persist()
    logging.info('time to persist : %s seconds',time.time()-t_pers)
    ds = do_inferences_from_slc_xspectra(subset,datedt,satellite,model,kx_ori,ky_ori,crossSpectraRe,crossSpectraIm,
                                    use_cwave_lib_without_cartpolconv=False)

    return ds

def treat_xspectra_fullimage(allspecs,satellite,datedt,slc,model,use_cwave_lib_without_cartpolconv=False):
    """

    :param allspecs:
    :param satellite:
    :param datedt:
    :param slc:
    :param model:
    :param use_cwave_lib_without_cartpolconv:
    :return:
    """
    kx_ori = allspecs['kx'].values
    ky_ori = allspecs['ky'].values
    crossSpectraRe = np.abs(allspecs['cross-spectrum_2tau'].mean(dim='2tau').real)
    crossSpectraIm = np.abs(allspecs['cross-spectrum_2tau'].mean(dim='2tau').imag)
    dsinfFullIm = do_inferences_from_slc_xspectra(slc,datedt,satellite,
                                                   model=model,kx_ori=kx_ori
                                                   ,ky_ori=ky_ori,
                                                   crossSpectraRe=crossSpectraRe,
                                                   crossSpectraIm=crossSpectraIm,
                                                   use_cwave_lib_without_cartpolconv=use_cwave_lib_without_cartpolconv)
    return dsinfFullIm



def get_data_for_inferences_on_subdomains(onetiff):
    """

    :param onetiff: (str) full path
    :return:
    """
    # step 1 read DN and compute sigma0 in tiff
    slc = xsarsea.cross_spectra_core.read_slc(onetiff,slice_subdomain=None) # use a specific image with DN matrix size matching sigma0 xsar size

    t0 = time.time()
    satellite = os.path.basename(onetiff)[0:3]

    datedt = datetime.datetime.strptime(os.path.basename(onetiff).split('-')[5],'%Y%m%dt%H%M%S')
    allspecs = None
    allspecs_per_sub_domain = None
    # step 2 compute cross spectra on each sub images
    subimage_range=512
    subimage_azimuth=512
    #subimage_range=1024
    #subimage_azimuth=1024
    #subimage_range = 2048
    #subimage_azimuth = 2048
    overlap_size = 256
    #overlap_size = 0
    allspecs,frange,fazimuth,allspecs_per_sub_domain,splitting_image,limits_sub_images = \
        xsarsea.cross_spectra_core.compute_SAR_cross_spectrum(slc,
                                    N_look=3,look_width=0.25,
                                    look_overlap=0.,look_window=None,range_spacing=None,
                                   welsh_window='hanning',
                                    nperseg={'range' : subimage_range,'azimuth' : subimage_azimuth},
                                   noverlap={'range' : overlap_size,'azimuth' : overlap_size}
                                    ,spacing_tol=1e-3)
    logging.info('time to get %s X-spectra : %1.1f seconds',len(allspecs_per_sub_domain),time.time()-t0)
    logging.info('loading model')
    modelQuach2020 = load_quach_2020_keras_model.load_quach2020_model_v2()
    # step 3 for each sub images compute cwaves params and NRCS and Nv
    all_ds_subs = []
    for subdomain in allspecs_per_sub_domain:
        t_subdo = time.time()
        ds_sub_im = treat_xspectra_one_subimage(allspecs_per_sub_domain,subdomain,satellite,datedt,slc,
                                                splitting_image,model=modelQuach2020)
        if subdomain%1==0:
            logging.info("inferences on sub images %i/%s elapsed time %1.1f seconds hs: %1.3fm",
                         subdomain,len(allspecs_per_sub_domain),time.time()-t_subdo,ds_sub_im['HsQuach'].values)
        all_ds_subs.append(ds_sub_im)
    all_ds_subs = xarray.concat(all_ds_subs,dim='subdomDim')
    all_ds_subs = xarray.merge([all_ds_subs,limits_sub_images]) #add the limits of the sub images
    # step 7 save the variables in a netCDF
    outputfile = '/home1/scratch/agrouaze/prediction_hs_over_sub_SAR_images_%s_%sx%s_v2.nc'%(os.path.basename(onetiff).replace('.tiff',''),subimage_azimuth,subimage_range)
    if os.path.exists(outputfile):
        os.remove(outputfile)
        logging.info('remove existing file %s',outputfile)
    all_ds_subs.attrs['subimage_range']=subimage_range
    all_ds_subs.attrs['subimage_azimuth']=subimage_azimuth
    all_ds_subs.to_netcdf(outputfile)
    logging.info('successfully creation written of output file : %s',outputfile)
    #save the file for the full image:
    ds_inferences_full_image = treat_xspectra_fullimage(allspecs,satellite,datedt,slc,model=modelQuach2020,
                                                        use_cwave_lib_without_cartpolconv=False)
    outputfile_full = '/home1/scratch/agrouaze/prediction_hs_over_full_SAR_images_%s_%sx%s_v3.nc' % (
    os.path.basename(onetiff).replace('.tiff',''),subimage_azimuth,subimage_range)
    ds_inferences_full_image.to_netcdf(outputfile_full)
    logging.info('successfully creation written of output file : %s',outputfile_full)
    return outputfile,outputfile_full

if __name__ =='__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='inferences Hs on SLC x spectra')
    parser.add_argument('--verbose',action='store_true',default=False)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')

    logging.debug('test')
    t0 = time.time()
    one_tiff = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2019/278/S1A_WV_SLC__1SSV_20191005T163939_20191005T171407_029326_035559_2461.SAFE/measurement/s1a-wv1-slc-vv-20191005t165023-20191005t165026-029326-035559-045.tiff'
    # one image with same size in tiff and setninel gdal driver
    #one_tiff = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2019/278/S1A_WV_SLC__1SSV_20191005T163939_20191005T171407_029326_035559_2461.SAFE/measurement/s1a-wv1-slc-vv-20191005t170433-20191005t170436-029326-035559-103.tiff'
    logging.info('file to treat : %s',os.path.basename(one_tiff))
    get_data_for_inferences_on_subdomains(one_tiff)
    logging.info('done in %s seconds',time.time()-t0)