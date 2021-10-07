"""
exp 2: dataset D4 :  5 cross spectra nouguier on 5 sub domains of the WV image, Hs CCI alti, NRCS, Nv and date of year/lon/lat, no CWAVE
Sept 2021
A Grouazel
inspiration rebuild_training_dataset_exp1.py
tested with cwave
INPUTS: SLC WV, alti CCI Hs,
analysis done in 16752.573705911636 seconds -> presque 5 heures
13/09/2021 07:35:35 INFO  peak memory usage: 49420.0546875 Mbytes -> presque 50Go !!!!!

algo:
1) read coloc CCI alti vs CCI SAR -> return tiff fullpath and Hs variables from alti
2) read SLC tiff -> get NRCS , Nv, cross spectre total + sub images x5
3) save all the params in a netCDF file (daily for instance one per SAr unit and per alti mission)
"""
import os
import sys
sys.path.append('/home1/datahome/agrouaze/git/xsarseafork/src/')
sys.path.append('/home1/datahome/agrouaze/git/xsarseafork/src/xsarsea')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn')
import logging
import xarray
import numpy as np
from scipy import interpolate
import datetime
import glob
#import xsarsea
import xsarsea.cross_spectra_core
import cross_spectra_core_dev_pyfftw
import warnings
import copy
import time
import spectrum_clockwise_to_trigo #xsarsea
import spectrum_rotation #xsarsea
from collections import defaultdict
import get_full_path_from_measurement
from sarhspredictor.lib.comp_xspec_on_5_sub_domains import prepare_image_splitting_in_5_domains,comp_xspec_for_one_of_the_5_subimages
from sarhspredictor.lib.compute_CWAVE_params import format_input_CWAVE_vector_from_OCN
from sarhspredictor.lib.predict_hs_from_SLC import compute_Cwave_params_and_xspectra_fromSLC
import resource
warnings.simplefilter(action='ignore',category=FutureWarning)
from scipy.spatial import KDTree
import traceback
import pdb
#CCI_ALTI_MISSION = ['al','s3b','s3a','cfo','j3','c2']
POSSIBLES_CCI_ALTI = {'cryosat-2':'cryosat-2',
                      #'envisat':'ENVISAT',
                     #'jason-1':'Jason-3',
                     'jason-2':'Jason-2',
                     'jason-3':'Jason-3',
                     'saral':'SARAL'} # in v2.0.6
reference_oswK_1145m_60pts = np.array([0.005235988, 0.00557381, 0.005933429, 0.00631625, 0.00672377,
    0.007157583, 0.007619386, 0.008110984, 0.008634299, 0.009191379,
    0.0097844, 0.01041568, 0.0110877, 0.01180307, 0.01256459, 0.01337525,
    0.01423822, 0.01515686, 0.01613477, 0.01717577, 0.01828394, 0.01946361,
    0.02071939, 0.02205619, 0.02347924, 0.02499411, 0.02660671, 0.02832336,
    0.03015076, 0.03209607, 0.03416689, 0.03637131, 0.03871796, 0.04121602,
    0.04387525, 0.04670605, 0.0497195, 0.05292737, 0.05634221, 0.05997737,
    0.06384707, 0.06796645, 0.0723516, 0.07701967, 0.08198893, 0.08727881,
    0.09290998, 0.09890447, 0.1052857, 0.1120787, 0.1193099, 0.1270077,
    0.1352022, 0.1439253, 0.1532113, 0.1630964, 0.1736193, 0.1848211,
    0.1967456, 0.2094395])

DIR_ORIGINAL_COLOCS = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/cci_orbit_files/v3.2_colocations_CMEMS/'
DIR_ORIGINAL_COLOCS = '/home1/scratch/agrouaze/'
OUTPUTDIR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp2/training_dataset/v1/'


def read_coloc_file(sar_unit,date_day,alti_mission):
    """
    for a given day return the ds containing coloc bt CCI alti vs SAR alti (if there is one)
    :param slc:
    :param ocn:
    :return:
        new_times : nb array of datetime obj SAR dates without microseconds
    """
    ds_coloc = None
    coloc_file_cci_sar_alti = os.path.join(DIR_ORIGINAL_COLOCS,
                                 sar_unit.upper() +'_'+alti_mission,date_day.strftime('%Y'), 'coloc_CCI_' + date_day.strftime('%Y%m%d')
                                 +'_'+sar_unit+'_'+alti_mission+'_3_hours_2_degree.nc')
    logging.debug('potential coloc file: %s',coloc_file_cci_sar_alti)
    if os.path.exists(coloc_file_cci_sar_alti):
        logging.info('coloc file exists : %s',coloc_file_cci_sar_alti)
        logging.debug('coloc_file_JS : %s',coloc_file_cci_sar_alti)
        ds_coloc = xarray.open_dataset(coloc_file_cci_sar_alti)
        ds_coloc['lon_SAR'] = ds_coloc['lon_SAR'].persist()
        ds_coloc['lat_SAR'] = ds_coloc['lat_SAR'].persist()
    else:
        logging.info('no coloc file for %s %s %s',sar_unit,alti_mission,date_day)
    return ds_coloc

def get_tiff_path(ds_coloc,sar_unit):
    """

    :param ds_coloc: xarray dataset
    :param sar_unit: str S1A or S1B
    :return:
    """
    all_tiff_fp = []
    for ii in range(len(ds_coloc['time_sar'].values)):
        ts = (ds_coloc['time_sar'].values[ii] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
        dt = datetime.datetime.utcfromtimestamp(ts)
        fptiff = get_full_path_from_measurement.get_full_path_ocn_wv_from_approximate_date(dt,sar_unit,level='L1')
        all_tiff_fp.append(fptiff)
    return all_tiff_fp


def store_cart_xspec_Ntau(all_computed_cart_xspectrum,subsetcoloc,nperseg,prefix='',which_tau=[0,1,2]):
    """

    :param all_computed_cart_xspectrum:
    :param subsetcoloc:
    :param nperseg:
    :param prefix: str
    :return:
    """
    for tau_n in which_tau :

        # for n_spec in range(len(allspecs['cross-spectrum_%stau'%tau_n])):
        # interpolate the cartesian cross spectra on a fixed grid
        # fixed_kx = np.arange(-0.39349586,0.39311158,0.00038427)
        fixed_kx = np.linspace(-0.39349586,0.39311158,nperseg['range'])
        fixed_ky = np.linspace(-0.7596674,0.75892554,nperseg['azimuth'])
        x = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n]['kx'].values
        y = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n]['ky'].values
        z_re = np.abs(all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n].mean(dim='%stau' % tau_n).real)
        f_re = interpolate.interp2d(x,y,z_re.values,kind='linear')
        z_re_new = f_re(fixed_kx,fixed_ky)
        subsetcoloc[prefix+'crossSpectraReCart_tau%s' % (tau_n)] = xarray.DataArray(z_re_new,dims=['kx','ky'],
                                                                             coords={'kx' : fixed_kx,
                                                                                     'ky' : fixed_ky})
        if tau_n != 0 :  # imag part of co spectre
            z_im = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n].mean(dim='%stau' % tau_n).imag
            f_im = interpolate.interp2d(x,y,z_im.values,kind='linear')
            z_im_new = f_im(fixed_kx,fixed_ky)
            subsetcoloc[prefix+'crossSpectraImCart_tau%s' % (tau_n)] = xarray.DataArray(z_im_new,dims=['kx','ky'],
                                                                                 coords={'kx' : fixed_kx,
                                                                                         'ky' : fixed_ky})
    return subsetcoloc


def store_pol_xspec_Ntau(all_computed_cart_xspectrum,platform_heading,nperseg,prefix='',apply_rotations=True):
    """
    1) interpolate the cartesian x spectra on a common grid cartesian
    2) interpolate the cartesian x spectra on a polar grid
    :param all_computed_cart_xspectrum:
    :param nperseg:
    :param prefix: str
    :return:
    """
    for tau_n in range(3) :

        # for n_spec in range(len(allspecs['cross-spectrum_%stau'%tau_n])):
        # interpolate the cartesian cross spectra on a fixed grid
        # fixed_kx = np.arange(-0.39349586,0.39311158,0.00038427)
        fixed_kx = np.linspace(-0.39349586,0.39311158,nperseg['range'])
        fixed_ky = np.linspace(-0.7596674,0.75892554,nperseg['azimuth'])
        x = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n]['kx'].values
        y = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n]['ky'].values
        z_re = np.abs(all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n].mean(dim='%stau' % tau_n).real)
        #z_re.assign_coords({'kx':fixed_kx,'ky':fixed_ky})
        f_re = interpolate.interp2d(x,y,z_re.values,kind='linear')
        z_re_new = f_re(fixed_kx,fixed_ky)
        z_re_new_da = xarray.DataArray(z_re_new,dims=['kx','ky'],
                         coords={'kx' : fixed_kx,
                                 'ky' : fixed_ky})
        crossSpectraRePol = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(z_re_new_da,Nphi=72,
                                                                                       ksampling='log',
                                                                                       **{'Nk' : 60,'kmin' :
                                                                                           reference_oswK_1145m_60pts[
                                                                                               0],'kmax' :
                                                                                              reference_oswK_1145m_60pts[
                                                                                                  -1]})
        if apply_rotations :
            crossSpectraRePol = spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(
                crossSpectraRePol)
            crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol,
                                                                 90.)  # This is for having origin at North
            crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol,platform_heading)
        #subsetcoloc[prefix+'crossSpectraRePol_tau%s' % (tau_n)] = crossSpectraRePol
        if tau_n != 0 :  # imag part of co spectre
            z_im = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n].mean(dim='%stau' % tau_n).imag
            #z_im.assign_coords({'kx' : fixed_kx,'ky' : fixed_ky})
            f_im = interpolate.interp2d(x,y,z_im.values,kind='linear')
            z_im_new = f_im(fixed_kx,fixed_ky)
            z_im_new_da = xarray.DataArray(z_im_new,dims=['kx','ky'],
                                                                                 coords={'kx' : fixed_kx,
                                                                                         'ky' : fixed_ky})
            crossSpectraImPol = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(z_im_new_da,Nphi=72,
                                                                                           ksampling='log',
                                                                                           **{'Nk' : 60,'kmin' :
                                                                                               reference_oswK_1145m_60pts[
                                                                                                   0],'kmax' :
                                                                                                  reference_oswK_1145m_60pts[
                                                                                                      -1]})
            if apply_rotations:
                crossSpectraImPol = spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(
                    crossSpectraImPol)
                crossSpectraImPol = spectrum_rotation.apply_rotation(crossSpectraImPol,
                                                                     90.)  # This is for having origin at North
                crossSpectraImPol = spectrum_rotation.apply_rotation(crossSpectraImPol,platform_heading)
            #subsetcoloc[prefix + 'crossSpectraImPol_tau%s' % (tau_n)] = crossSpectraImPol
            # subsetcoloc[prefix+'crossSpectraImPol_tau%s' % (tau_n)] = xarray.DataArray(z_im_new,dims=['kx','ky'],
            #                                                                      coords={'kx' : fixed_kx,
            #                                                                              'ky' : fixed_ky})
    return crossSpectraRePol,crossSpectraImPol

def read_all_SAR_variables(sar_unit,date_day,alti_mission,dev=False,grid_xspec='polar'):
    """

    :param sar_unit:
    :param date_day: datetime
    :param alti_mission:
    :param dev:
    :param grid_xspec: str (polar or cartesian)
    :return:
    """
    tau_to_keep = [2]
    ds_coloc = read_coloc_file(sar_unit,date_day,alti_mission)
    list_tiff_paths = get_tiff_path(ds_coloc,sar_unit)
    logging.debug('Nb tiff path: %s while expected :%s',len(list_tiff_paths),ds_coloc['lon_SAR'].size)
    list_base_tiff = [os.path.basename(hh) for hh in list_tiff_paths]
    ds_coloc['tiff'] = xarray.DataArray(np.array(list_base_tiff),dims=['time_sar']) #,coords={'time_sar':ds_coloc['time_sar']}

    nb_match = len(list_tiff_paths)
    all_subsets_coloc = []
    #for xxx,indxs in enumerate(inds):
    for xxx in range(nb_match):
        one_tiff = list_tiff_paths[xxx]
        logging.info('prepare coloc : %s/%s',xxx+1,nb_match)
        # 1) read all params from Justin s dataset
        subsetcoloc = ds_coloc.isel(time_sar=np.array([xxx])) #{ on peut selectionner plusieurs indice en meme temps avec isel}

        logging.info('subsetcoloc : %s',subsetcoloc)
        #TODO: voir si on peut ajouter ces variables dans les colocs pour ne pas avoir a les calculer sur le tiff
        #ta = subsetcoloc['trackAngle'].values[0]
        xsarslc = cross_spectra_core_dev_pyfftw.read_slc(one_tiff,slice_subdomain=None,resolution=None,resampling=None)
        logging.debug('xsarslc : %s',xsarslc)
        ta = xsarslc.attrs['heading']
        s0 = np.nanmean(xsarslc['sigma0'].values)
        #s0 = subsetcoloc['sigma0'].values[0]
        #nv = subsetcoloc['normalizedVariance'].values[0]
        nv = np.nanvar(xsarslc['sigma0'].values,ddof=1) / s0**2 #np.var(dummy, ddof=1) / intensity ** 2.
        logging.debug('nv : %s',nv)
        #incidenceangle = subsetcoloc['incidenceAngle'].values[0]
        incidenceangle = xsarslc['incidence'].values.mean()
        size_az,size_ra = xsarslc['longitude'].shape
        mid_range_ind = int(size_az/2)
        mid_azi_ind = int(size_ra/2)
        lons_all = xsarslc['longitude'].values
        lats_all = xsarslc['latitude'].values
        lonsar = lons_all[mid_azi_ind,mid_range_ind]
        latsar = lats_all[mid_azi_ind,mid_range_ind]
        logging.debug('lonsar : %s latsar : %s',lonsar,latsar)
        subsetcoloc['sigma0'] = xarray.DataArray([10.*np.log10(s0)],dims=['time_sar'])
        subsetcoloc['sigma0'].attrs = {'description':'mean sigma0 denoised from SLC WV read by xsar',
                                     'unit':'dB',
                                     'longname':'normalized radar cross section'}

        subsetcoloc['normalized_variance'] = xarray.DataArray([nv],dims=['time_sar'])
        subsetcoloc['normalized_variance'].attrs = {'description' : 'normalized variance from image SLC WV intensity read by xsar',
                                     'longname' : 'normalized variance of radar cross section'}

        subsetcoloc['incidence_angle'] = xarray.DataArray([incidenceangle],dims=['time_sar'])
        subsetcoloc['incidence_angle'].attrs = {'description' : 'mean incidence angle of SLC WV image',
                                              'unit' : 'deg',
                                     'longname' : 'incidence angle'}

        subsetcoloc['track_angle'] = xarray.DataArray([ta],dims=['time_sar'])
        subsetcoloc['track_angle'].attrs = {'description' : 'angle between satellite orbit and North, clockwise',
                                     'longname' : 'platform track angle',
                                          'unit':'deg',
                                          'note':'platform track angle can be different from local bearinig angle in SAR image'}
        #same operation bu using level1 informations
        dsslc = xsarsea.cross_spectra_core.read_slc(one_tiff)
        dsslc = dsslc.sel(pol='VV')
        if dev :
            nperseg = {'range' : 2048,'azimuth' : 2048}
            nperseg = {'range' : 1024,'azimuth' : 1024}
            nperseg = {'range' : 512,'azimuth' : 512}
            nperseg = {'range' : 700,'azimuth' : 700}
        else :
            nperseg = {'range' : 512,'azimuth' : 512}
        t0 = time.time()
        allspecs,frange,fazimuth,allspecs_per_sub_domain,splitting_image, \
        limits_sub_images = cross_spectra_core_dev_pyfftw.compute_SAR_cross_spectrum(
            dsslc,
            N_look=3,look_width=0.25,
            look_overlap=0.,look_window=None,  # range_spacing=slc.attrs['rangeSpacing']
            welsh_window='hanning',
            nperseg=nperseg,
            noverlap={'range' : 256,'azimuth' : 256}
            ,spacing_tol=1e-3,debug_plot=False,fft_version='pyfftw',return_periodoXspec=True)
        logging.info('time to get %s X-spectra : %1.1f seconds',len(splitting_image),time.time() - t0)
        # 3) interpolate and convert cartesian grid to polar 72,60
        #xspecReCart = np.abs(allspecs['cross-spectrum_2tau'].mean(dim='2tau').real)
        if grid_xspec=='cartesian':
            subsetcoloc = store_cart_xspec_Ntau(allspecs,subsetcoloc,nperseg,which_tau=tau_to_keep)
        elif grid_xspec=='polar':
            xsspecCross_Polar_Re_fullspan,xsspecCross_Polar_Im_fullspan = store_pol_xspec_Ntau(allspecs,
                                                                        nperseg=nperseg,platform_heading=ta
                                                                        ,which_tau=tau_to_keep)
            subsetcoloc['xsspecCross_Polar_Re_fullspan'] = xsspecCross_Polar_Re_fullspan
            subsetcoloc['xsspecCross_Polar_Im_fullspan'] = xsspecCross_Polar_Im_fullspan
        else:
            raise Exception('unknown grid x spec %s'%grid_xspec)

        #add spectrum per sub domain
        Re_subs = None
        Im_subs = None
        for rect_id in allspecs_per_sub_domain :
            if grid_xspec=='cartesian':
                subsetcoloc = store_cart_xspec_Ntau(allspecs_per_sub_domain[rect_id],subsetcoloc,nperseg,prefix=str(rect_id)+'-')
            elif grid_xspec=='polar':
                xsspecCross_Polar_Re_subtmp,xsspecCross_Polar_Im_subtmp = store_pol_xspec_Ntau(allspecs_per_sub_domain[rect_id],
                                                nperseg=nperseg,platform_heading=ta,prefix='dom'+str(rect_id)+'-')
                if Re_subs is None:
                    Re_subs = xsspecCross_Polar_Re_subtmp
                    Im_subs = xsspecCross_Polar_Im_subtmp
                else:
                    Re_subs = xarray.concat([Re_subs,xsspecCross_Polar_Re_subtmp],dim='sub_domain')
                    Im_subs = xarray.concat([Im_subs,xsspecCross_Polar_Im_subtmp],dim='sub_domain')
            else:
                raise Exception('unknown grid x spec %s'%grid_xspec)
        subsetcoloc['xspec_polar_Re_sub_domains'] = Re_subs
        subsetcoloc['xspec_polar_Im_sub_domains'] = Im_subs
        subsetcoloc.assign_coords({'sub_domain':np.arange(len(allspecs_per_sub_domain))})
        # add geolocations of the subdomains
        # store long/lat of each rectangles
        geoloc = {}
        for rect_x in splitting_image :
            sli_az,sli_ra = splitting_image[rect_x]['azimuth'],splitting_image[rect_x]['range']
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
            if 'lons' not in geoloc:
                geoloc = {'lons' : np.array(lons),'lats' : np.array(lats)}
            else:
                geoloc['lons'] = np.vstack([geoloc['lons'],np.array(lons)])
                geoloc['lats'] = np.vstack([geoloc['lats'],np.array(lats)])
        logging.info('rects lons : %s',geoloc['lons'].shape)
        subsetcoloc['lons_rects'] = xarray.DataArray(geoloc['lons'],dims=['nb_domains','dim_poly'],
                                        coords={'dim_poly' : np.arange(5),'nb_domains':np.arange(len(splitting_image))})
        subsetcoloc['lats_rects'] = xarray.DataArray(geoloc['lats'],dims=['nb_domains','dim_poly'],
                                        coords={'dim_poly' : np.arange(5),'nb_domains':np.arange(len(splitting_image))})

        # I add the 5 sub xspectra
        # ici je fais le choix d avoir 5 sous domains de 512x512 (sans sub division dans les sous domaine)
        # split_inds_dict,geolocs_rect = prepare_image_splitting_in_5_domains(xsarslc,limited_width_domains=nperseg['range'])
        # for rect_id in geolocs_rect:
        #     subsetcoloc['lons_%s'%rect_id] = xarray.DataArray(geolocs_rect[rect_id]['lons'],dims=['dim_poly'],coords={'dim_poly':np.arange(5)})
        #     subsetcoloc['lats_%s' % rect_id] = xarray.DataArray(geolocs_rect[rect_id]['lats'],dims=['dim_poly'],
        #                                                         coords={'dim_poly' : np.arange(5)})
        # for di,onesubim in enumerate(split_inds_dict) :
        #     sli_az_range = split_inds_dict[onesubim]
        #     logging.info('%s %s : %s',di,onesubim,sli_az_range)
        #
        #     allspecs_one_out_of_5_domains = comp_xspec_for_one_of_the_5_subimages(one_tiff,
        #                                                      slice_im_az_range=sli_az_range)
        #     if grid_xspec == 'cartesian' :
        #         subsetcoloc = store_cart_xspec_Ntau(allspecs_one_out_of_5_domains,subsetcoloc,nperseg,prefix=onesubim+'-')
        #     elif grid_xspec == 'polar' :
        #         subsetcoloc = store_pol_xspec_Ntau(allspecs_one_out_of_5_domains,subsetcoloc,nperseg=nperseg,
        #                                            platform_heading=ta,prefix=onesubim+'-')
        #     else :
        #         raise Exception('unknown grid x spec %s' % grid_xspec)


        #logging.info('crossSpectraRePol %s',crossSpectraRePol_xa.shape)
        logging.info('ta : %s',ta)
        logging.info('incidenceangle : %s',incidenceangle)
        # lstvars_with_scale_factor_and_offset = ['hsALTmin','hsALTmax','incidenceAngle','hsALT','hsWW3','wsALTmin',
        #                                         'wsALT','wsALTmax','dx','dt','nk','nth','hsSM','h200','h400','h800',
        #                                         'trackAngle','hsWW3v2']
        # for vvy in lstvars_with_scale_factor_and_offset :
        #     if vvy in subsetcoloc:
        #         subsetcoloc[vvy].encoding = {}
        #subsetcoloc = subsetcoloc.drop('k')  # to avoid ambiguous k coordinates definition
        # for hh in subsetcoloc :
        #     if 'prb' in hh :
        #         subsetcoloc = subsetcoloc.drop(hh)

        #subsetcoloc['py_S'] = xarray.DataArray(np.tile(S_slc.T,(1,1)),dims=['time','N'],coords={'time' : times_bidons,'N' : np.arange(20)})
        #subsetcoloc['py_S'].attrs['description'] = 'S params from SLC xspectra'
        #subsetcoloc['S'] = xarray.DataArray(np.tile(Socn.T,(1,1)),dims=['time','N'],coords={'time' : times_bidons,'N' : np.arange(20)})
        #subsetcoloc['S'].attrs['description'] = 'S params from OCN xpectra'
        #subsetcoloc['py_S'] = xarray.DataArray(S.T,dims=['time','N'],coords={'time':[datedt_slc],'N':np.arange(20)}) #solution simple
        #subsetcoloc['py_S'] = subsetcoloc['py_S'].attrs['description']='20 C-WAVE params computed from polar cross spectra 2-tau'
        all_subsets_coloc.append(subsetcoloc)
        if dev and xxx==1:
            logging.info('break the loop over the matchups after 2 iterations with dev/test option')
            break
    return all_subsets_coloc

def save_training_file(dscoloc_enriched,outputfile):
    """

    :param dscoloc_enriched: contains py_S and X-spectra from SLC + Hs altimetric
    :param outputfile:
    :return:
    """
    # 5 ) save a netcdf file
    dscoloc_enriched.attrs['created_on'] = '%s' % datetime.datetime.today()
    dscoloc_enriched.attrs['created_by'] = 'Antoine Grouazel'
    dscoloc_enriched.attrs['purpose'] = 'SAR Hs NN regression exp#2'
    dscoloc_enriched.attrs['purpose'] = 'content SAR & Alti colocations prepared using CCI L2p dataset v2.0.6'
    dscoloc_enriched.to_netcdf(outputfile)
    logging.info('outputfile : %s',outputfile)
    os.chmod(outputfile,0o0777)
    logging.info('set permission 777 on output file done')


if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser(description='prepare_training_Hs_NN_dataset')
    parser.add_argument('--verbose',action='store_true',default=False)
    parser.add_argument('--dev',action='store_true',default=False,required=False,
                        help='dev mode with reduced number of periodograms (size 2048 instead of 512)')
    parser.add_argument('--redo',action='store_true',default=False,required=False,
                        help='redo existing files')
    parser.add_argument('--date',action='store',required=True,help='date YYYYMMDD')
    parser.add_argument('--sar-unit',action='store',required=True,help='S1A or S1B or ... ')
    parser.add_argument('--alti',choices=POSSIBLES_CCI_ALTI.keys(),required=True,help='alti mission in %s'%(POSSIBLES_CCI_ALTI.keys()))
    parser.add_argument('--outputdir',action='store',default=OUTPUTDIR,required=False,help='outputdir [optional, default is %s]'%OUTPUTDIR)
    args = parser.parse_args()
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    t1 = time.time()
    time.sleep(np.random.randint(0,5,1)[0]) # to avoid mkdir issues with p-run
    #slc = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2015/017/S1A_WV_SLC__1SSV_20150117T124852_20150117T130516_004211_0051DB_E791.SAFE/measurement/s1a-wv2-slc-vv-20150117t125754-20150117t125757-004211-0051db-038.tiff'
    #ocn = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2015/017/S1A_WV_OCN__2SSV_20150117T130513_20150117T130516_004211_0051DB_0852.SAFE/measurement/s1a-wv1-ocn-vv-20150117t124852-20150117t130517-004211-0051DB-053.nc'
    #slc = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1b/L1/WV/S1B_WV_SLC__1S/2018/197/S1B_WV_SLC__1SSV_20180716T174520_20180716T180835_011839_015CA3_AA8D.SAFE/measurement/s1b-wv1-slc-vv-20180716t180521-20180716t180524-011839-015ca3-083.tiff'
    #ocn = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1b/L2/WV/S1B_WV_OCN__2S/2018/197/S1B_WV_OCN__2SSV_20180716T174520_20180716T180835_011839_015CA3_D1EE.SAFE/measurement/s1b-wv1-ocn-vv-20180716t180521-20180716t180524-011839-015ca3-083.nc'
    # direct matching multi indices
    #slc = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L1/WV/S1A_WV_SLC__1S/2018/001/S1A_WV_SLC__1SSV_20180101T132025_20180101T134211_019961_021FEA_C3D7.SAFE/measurement/s1a-wv2-slc-vv-20180101t132040-20180101t132043-019961-021fea-002.tiff'
    #ocn = '/home/datawork-cersat-public/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2018/001/S1A_WV_OCN__2SSV_20180101T132025_20180101T134211_019961_021FEA_7EBF.SAFE/measurement/s1a-wv2-ocn-vv-20180101t132040-20180101t132043-019961-021fea-002.nc'
    if args.outputdir:
        outdir = args.outputdir
    else:
        outdir = OUTPUTDIR
    datedt = datetime.datetime.strptime(args.date,'%Y%m%d')
    outputfile = os.path.join(outdir,datedt.strftime('%Y'),
                              datedt.strftime('%j'),'training_D4_exp2_%s_%s_%s.nc' %(args.date,args.alti,args.sar_unit))
    logging.info('outputfile : %s',outputfile)
    if os.path.exists(os.path.dirname(outputfile)) is False :
        os.makedirs(os.path.dirname(outputfile),0o0775)
    # remove is needed

    if os.path.exists(outputfile) and args.redo:
        os.remove(outputfile)
    # skip if already present (on ne sais pas a l avance combien il a de matching)
    if os.path.exists(outputfile) and args.redo is False:
        logging.info('nothing to do, the file already exists')
        sys.exit(0)
    else:
        date_to_treat_dt = datetime.datetime.strptime(args.date,'%Y%m%d')
        list_dscoloc_enriched = read_all_SAR_variables(sar_unit=args.sar_unit,alti_mission=args.alti,
                                                       date_day=date_to_treat_dt,dev=args.dev,grid_xspec='polar')
        #dscoloc_enriched = xarray.merge(list_dscoloc_enriched) # here merge and concat are doing bullshit (increase kx and ky size...) -> I do it by myself
        #before concat I need to align number of subdomains (on the minimum avail)
        min_subdom_nb = 10000
        for pp in list_dscoloc_enriched:
            val_subdom = pp['sub_domain'].size
            if val_subdom<min_subdom_nb:
                min_subdom_nb = val_subdom
        logging.info('min sub domain: %s',min_subdom_nb)
        new_list_ds = []
        for pp in list_dscoloc_enriched:
            pp2 = pp.isel({'sub_domain':slice(0,min_subdom_nb)})
            new_list_ds.append(pp2)
        dscoloc_enriched = xarray.concat(new_list_ds,dim='time_sar')
        # data_cat = {}
        # for onematch in list_dscoloc_enriched:
        #     for vv in onematch:
        #         if vv not in data_cat:
        #             data_cat[vv] = onematch[vv]
        #         else:
        #             data_cat[vv] = xarray.concat(data_cat[vv],dim='time_sar')
        #dscoloc_enriched = xarray.Dataset(data_cat)
        save_training_file(dscoloc_enriched,outputfile)
    logging.info('analysis done in %s seconds',time.time()-t1)
    logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)