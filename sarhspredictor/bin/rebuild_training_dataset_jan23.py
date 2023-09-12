"""
dataset janver 2023:  cartesian cross spectra xsar_slc avec correction RI of the WV image, Hs CMEMS+CCi alti, NRCS, Nv and date of year/lon/lat, and CWAVE
Janvier 2023
A Grouazel

algo:
1) read coloc CCI alti vs CCI SAR -> return tiff fullpath and Hs variables from alti
2) read SLC tiff -> get NRCS , Nv, cross spectre total (+ sub images x5)
3) save all the params in a netCDF file (daily for instance one per SAR unit and per alti mission)

env to test: xsar_pr46
"""
import os
import sys
sys.path.append('/home1/datahome/agrouaze/git/mpc/data_collect')
# sys.path.append('/home1/datahome/agrouaze/git/xsarseafork/src/')
# sys.path.append('/home1/datahome/agrouaze/git/xsarseafork/src/xsarsea')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn')
import logging
import xarray
import numpy as np
from scipy import interpolate
import datetime
import glob
import xsar
# import conversion_polar_cartesian
import warnings
import copy
import time
# import spectrum_clockwise_to_trigo #xsarsea
# import spectrum_rotation #xsarsea
from collections import defaultdict
import get_full_path_from_measurement
from xsarslc.processing.xspectra import compute_WV_intraburst_xspectra
from xsarslc.processing import xspectra
from dspec.spectrum_momentum import computeOrthogonalMoments # CWAVE like but corrected for filter
#from sarhspredictor.lib.comp_xspec_on_5_sub_domains import prepare_image_splitting_in_5_domains,comp_xspec_for_one_of_the_5_subimages
#from sarhspredictor.lib.compute_CWAVE_params import format_input_CWAVE_vector_from_OCN
from sarhspredictor.lib.compute_CWAVE_params_from_cart_Xspectra import format_input_CWAVE_vector_from_SLC

KMAX_CWAVE = 2 * np.pi / 60
KMIN_CWAVE = 2 * np.pi / 625
from shared_information import DIR_L2F_WV_DAILY
# from cut_off import get_cut_off_profile # deja dans xsar_slc je crois
import resource
warnings.simplefilter(action='ignore',category=FutureWarning)
import traceback
import pdb
#CCI_ALTI_MISSION = ['al','s3b','s3a','cfo','j3','c2']
POSSIBLES_CCI_ALTI = {'cryosat-2':'cryosat-2',
                      #'envisat':'ENVISAT',
                     #'jason-1':'Jason-3',
                     'jason-2':'Jason-2',
                     'jason-3':'Jason-3',
                     'saral':'SARAL'} # in v2.0.6
POSSIBLES_CMEMS_ALTI = {'cryosat-2':'c2',
                      #'envisat':'ENVISAT',
                     #'jason-1':'Jason-3',
                        'cfosat':'cfo',
                     'jason-2':'j2',
                     'jason-3':'j3',
                     'saral':'al'}
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

DIR_ORIGINAL_COLOCS = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/cci_orbit_files/v3.2_colocations_CMEMS_v7/'

# prun->pbs already provide the outputdir
OUTPUTDIR = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/dataset_jan23/'

def read_coloc_file(sar_unit,date_day,alti_mission):
    """
    for a given day return the ds containing coloc bt CCI alti vs SAR alti (if there is one)
    :param slc:
    :param ocn:
    :return:
        new_times : nb array of datetime obj SAR dates without microseconds
    """
    t0 = time.time()
    ds_coloc = None
    coloc_file_cci_sar_alti = os.path.join(DIR_ORIGINAL_COLOCS,
                                 sar_unit.upper() +'_'+alti_mission,date_day.strftime('%Y'), 'coloc_' + date_day.strftime('%Y%m%d')
                                #sar_unit.upper() + '_' + alti_mission, date_day.strftime('%Y'), 'coloc_CCI_' + date_day.strftime('%Y%m%d')
                                 +'_'+sar_unit+'_'+alti_mission+'_3_hours_2_degree.nc')
    logging.info('potential coloc file: %s',coloc_file_cci_sar_alti)
    if os.path.exists(coloc_file_cci_sar_alti):
        logging.info('coloc file exists : %s',coloc_file_cci_sar_alti)
        logging.debug('coloc_file_JS : %s',coloc_file_cci_sar_alti)
        ds_coloc = xarray.open_dataset(coloc_file_cci_sar_alti)
        ds_coloc['lon_SAR'] = ds_coloc['lon_SAR'].persist()
        ds_coloc['lat_SAR'] = ds_coloc['lat_SAR'].persist()
    else:
        logging.info('no coloc file for %s %s %s',sar_unit,alti_mission,date_day)
    logging.info('time to read colocs : %s1.1f seconds',(time.time()-t0))
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
        logging.debug('dt : %s',dt)
        fptiff = get_full_path_from_measurement.get_full_path_ocn_wv_from_approximate_date(dt,sar_unit,level='L1')
        all_tiff_fp.append(fptiff)
    return all_tiff_fp


# def store_cart_xspec_Ntau(crossSpectraReCart2tau,crossSpectraImCart2tau,subsetcoloc,prefix=''):
#     """
#
#     :param crossSpectraReCart2tau: 2tau DataArray kx ky
#     :param crossSpectraImCart2tau: 2tau DataArray kx ky
#     :param subsetcoloc: xarray.Dataset
#     :param prefix: str
#     :return:
#     """
#
#     tau_n = 2 # hard coded
#     subsetcoloc[prefix+'crossSpectraReCart_tau%s' % (tau_n)] = crossSpectraReCart2tau
#     #if tau_n != 0 :  # imag part of co spectre
#
#     subsetcoloc[prefix+'crossSpectraImCart_tau%s' % (tau_n)] = crossSpectraImCart2tau
#     subsetcoloc['kx'].attrs['max'] = np.amax(subsetcoloc['kx'].values)
#     subsetcoloc['kx'].attrs['min'] = np.amin(subsetcoloc['kx'].values)
#     subsetcoloc['ky'].attrs['max'] = np.amax(subsetcoloc['ky'].values)
#     subsetcoloc['ky'].attrs['min'] = np.amin(subsetcoloc['ky'].values)
#     return subsetcoloc




# def average_cartesian_spec_2tau(all_computed_cart_xspectrum,nperseg,tau_n=2):
#     """
#
#     :param all_computed_cart_xspectrum: xarrayDataset
#     :param nperseg: dict
#     :param tau_n: int
#     :return:
#     """
#     t0 = time.time()
#     fixed_kx = np.linspace(-0.39349586, 0.39311158, nperseg['range'])
#     fixed_ky = np.linspace(-0.7596674, 0.75892554, nperseg['azimuth'])
#     x = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n]['kx'].values
#     y = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n]['ky'].values
#     z_re = np.abs(all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n].mean(dim='%stau' % tau_n).real)
#     # z_re.assign_coords({'kx':fixed_kx,'ky':fixed_ky})
#     f_re = interpolate.interp2d(x, y, z_re.values, kind='linear')
#     z_re_new = f_re(fixed_kx, fixed_ky)
#     z_re_new_da = xarray.DataArray(z_re_new, dims=['kx', 'ky'],
#                                    coords={'kx': fixed_kx,
#                                            'ky': fixed_ky})
#     z_im = all_computed_cart_xspectrum['cross-spectrum_%stau' % tau_n].mean(dim='%stau' % tau_n).imag
#     # z_im.assign_coords({'kx' : fixed_kx,'ky' : fixed_ky})
#     f_im = interpolate.interp2d(x, y, z_im.values, kind='linear')
#     z_im_new = f_im(fixed_kx, fixed_ky)
#     z_im_new_da = xarray.DataArray(z_im_new, dims=['kx', 'ky'],
#                                    coords={'kx': fixed_kx,
#                                            'ky': fixed_ky})
#     logging.info('time to re grid cartesian spec : %s1.1f seconds', (time.time() - t0))
#     return z_re_new_da,z_im_new_da

def add_WW3_spectra(date_sar,sar_unit):
    """
    TODO: mettre le spectre WW3 en cartesien avec rotation prealable pour le mettre dans l'orientation de limage SAR, a confirmer avec Alx et Fred
    :param date_sar: datetime
    :param sar_unit:  str S1A or ...
    :return:
    """
    daily_full_wv_pot = os.path.join(DIR_L2F_WV_DAILY,date_sar.strftime('%Y'),date_sar.strftime('%j'),sar_unit+'*.nc')
    effective_candidate = glob.glob(daily_full_wv_pot)
    ww3_xspec_polar = xarray.DataArray(np.ones((1, 60, 72)) * np.nan,
                                       dims=['time_sar', 'k', 'phi'],
                                       coords={'time_sar':[np.datetime64(date_sar)],'k':reference_oswK_1145m_60pts,'phi':np.radians(np.arange(0,360,5))},
                                       attrs={'description': 'WW3 spectra compute at the closest grid point ',
                                              'grid_spectra': 'polar',
                                              'model_integration_time': '1 hour',
                                              'grid resolution': '0.5 deg'})
    if len(effective_candidate)>0:
        daily_full_wv = effective_candidate[0]
        WVdaily_L2_IFR_ds = xarray.open_dataset(daily_full_wv)
        WVdaily_L2_IFR_ds['ww3_spec_interp72x60'] = WVdaily_L2_IFR_ds['ww3_spec_interp72x60'].persist()
        dates = WVdaily_L2_IFR_ds['fdatedt']
        indok = np.where(dates==np.datetime64(date_sar))[0]
        tmpWW3sp = WVdaily_L2_IFR_ds['ww3_spec_interp72x60'].isel(fdatedt=indok).values
        ww3_xspec_polar.data = np.swapaxes(tmpWW3sp,1,2)#[np.newaxis,:,:]
    else:
        pass
    return ww3_xspec_polar


def get_SAR_SLC_quantities(one_tiff,dev=False):
    """
    #TODO : voir si on genere des L1B d abord et ensuite on fait le training dataset ou bien on fait les 2 en meme temps.
    :param one_tiff: str
    :param dev: bool
    :return:
    """

    SLCWVds = xarray.Dataset()

    # TODO: voir si on peut ajouter ces variables dans les colocs pour ne pas avoir a les calculer sur le tiff
    # ta = subsetcoloc['trackAngle'].values[0]
    # xsarslc = cross_spectra_core_dev_pyfftw.read_slc(one_tiff,slice_subdomain=None,resolution=None,resampling=None)
    t_xsar = time.time()
    fullpathsafeSLC = os.path.dirname(os.path.dirname(one_tiff))
    imagette_number = os.path.basename(one_tiff).split('-')[-1].replace('.tiff', '')
    sar_unit = os.path.basename(one_tiff)[0:3].upper()
    date_sar_dt = datetime.datetime.strptime(os.path.basename(one_tiff).split('-')[4], '%Y%m%dt%H%M%S')
    #SLCWVds['time_sar'] = xarray.DataArray()
    ww3_da = add_WW3_spectra(date_sar=date_sar_dt, sar_unit=sar_unit)
    SLCWVds['ww3PolarElevationSpec'] = ww3_da
    str_gdal = 'SENTINEL1_DS:%s:WV_%s' % (fullpathsafeSLC, imagette_number)
    xsar_s1ds = xsar.Sentinel1Dataset(str_gdal)
    xsar_s1ds.add_high_resolution_variables()
    xsar_s1ds.apply_calibration_and_denoising()
    xsar_s1ds.datatree.load()
    xsarslc = xsar_s1ds.dataset
    logging.debug('xsarslc : %s', xsarslc)
    ta = xsarslc.attrs['platform_heading']
    s0 = xsarslc['sigma0'].mean().values
    # s0 = subsetcoloc['sigma0'].values[0]
    # nv = subsetcoloc['normalizedVariance'].values[0]
    nv = np.nanvar(xsarslc['sigma0'].values, ddof=1) / s0 ** 2  # np.var(dummy, ddof=1) / intensity ** 2.
    logging.debug('nv : %s', nv)
    # incidenceangle = subsetcoloc['incidenceAngle'].values[0]
    # incidenceangle = xsarslc['incidence'].values.mean()
    #incidenceangle = xsar_s1ds.s1meta.image['incidence_angle_mid_swath']
    incidenceangle = xsar_s1ds.datatree['image']['incidenceAngleMidSwath']
    size_az, size_ra = xsarslc['longitude'].shape
    mid_range_ind = int(size_az / 2)
    mid_azi_ind = int(size_ra / 2)
    # lon_centroid,lat_centroid = xsar_s1ds.footprint.centroid.xy
    # lons_all = xsarslc['longitude'].values
    # lats_all = xsarslc['latitude'].values
    # lonsar = lons_all[mid_azi_ind, mid_range_ind]
    # latsar = lats_all[mid_azi_ind, mid_range_ind]
    x_0, y_0 = np.asarray(xsar_s1ds.datatree['geolocation_annotation'].ds['longitude'].values.shape) // 2
    lonsar = xsar_s1ds.datatree['geolocation_annotation'].ds['longitude'].values[x_0,y_0]
    latsar = xsar_s1ds.datatree['geolocation_annotation'].ds['latitude'].values[x_0, y_0]
    SLCWVds['lonsar_SLC'] = xarray.DataArray([lonsar], dims=['time_sar'])
    SLCWVds['lonsar_SLC'].attrs = {'description': 'longitude at center of SLC WV image',
                                       'unit': 'deg',
                                       'range': (-180, 180),
                                       'longname': 'longitude'}
    SLCWVds['latsar_SLC'] = xarray.DataArray([latsar], dims=['time_sar'])
    SLCWVds['latsar_SLC'].attrs = {'description': 'latitude at center of SLC WV image',
                                       'unit': 'deg',
                                       'range': (-90, 90),
                                       'longname': 'latitude'}

    logging.debug('lonsar : %s latsar : %s', lonsar, latsar)
    SLCWVds['s0_SLC'] = xarray.DataArray([10. * np.log10(s0)], dims=['time_sar'])
    SLCWVds['s0_SLC'].attrs = {'description': 'mean sigma0 denoised from SLC WV read by xsar',
                                   'unit': 'dB',
                                   'longname': 'normalized radar cross section'}

    SLCWVds['nv_SLC'] = xarray.DataArray([nv], dims=['time_sar'])
    SLCWVds['nv_SLC'].attrs = {'description': 'normalized variance from image SLC WV intensity read by xsar',
                                   'longname': 'normalized variance of SLC image'}

    SLCWVds['incidenceangle_SLC'] = xarray.DataArray([incidenceangle], dims=['time_sar'])
    SLCWVds['incidenceangle_SLC'].attrs = {'description': 'mean incidence angle of SLC WV image',
                                               'unit': 'deg',
                                               'longname': 'incidence angle'}

    SLCWVds['ta_SLC'] = xarray.DataArray([ta], dims=['time_sar'])
    SLCWVds['ta_SLC'].attrs = {'description': 'angle between satellite orbit and North, clockwise',
                                   'longname': 'platform track angle',
                                   'alternative_name': 'heading angle / bearing angle',
                                   'unit': 'deg',
                                   'note': 'platform track angle can be different from local bearinig angle in SAR image'}
    # same operation but using level1 informations
    s1ds = xsar_s1ds.dataset
    #azimuthSpacing, rangeSpacing = xsar_s1ds.s1meta.image['ground_pixel_spacing']
    rangeSpacing = xsar_s1ds.dataset['sampleSpacing'].values
    azimuthSpacing = xsar_s1ds.dataset['lineSpacing'].values
    s1ds.attrs['rangeGroundSpacing'] = rangeSpacing
    s1ds.attrs['azimuthSpacing'] = azimuthSpacing

    logging.info('time to read data from SLC : %s1.1f seconds', (time.time() - t_xsar))
    t0 = time.time()
    all_computed_cart_xspectrum = compute_WV_intraburst_xspectra(dt=xsar_s1ds.datatree,
                                                                 polarization='VV',
                                                                 periodo_width={"line": 2000, "sample": 2000},
                                                                 periodo_overlap={"line": 1000, "sample": 1000})
    logging.info('time to get %s X-spectra : %1.1f seconds', all_computed_cart_xspectrum['2tau'], time.time() - t0)
    # 3) interpolate and convert cartesian grid to polar 72,60
    #

    xs = all_computed_cart_xspectrum.swap_dims({'freq_line': 'k_az', 'freq_sample': 'k_rg'})
    xs = xspectra.symmetrize_xspectrum(xs, dim_range='k_rg', dim_azimuth='k_az')

    ############################################ real part ############################
    z_re_new_da = np.abs(xs['xspectra_2tau'].mean(dim='2tau').real)
    z_im_new_da = xs['xspectra_2tau'].mean(dim='2tau').imag

    # je met ici le calcul des cwave à partir du SLC:
    t_cwave = time.time()
    spectralmoments = computeOrthogonalMoments(real_part_spectrum=z_re_new_da, imaginary_part_spectrum=z_im_new_da, kmax=KMAX_CWAVE, kmin=KMIN_CWAVE)
    logging.debug('shape CWAVE SLC %s', spectralmoments.shape)
    logging.info('time to compute CWAVE param from SLC: %s1.1f seconds', (time.time() - t_cwave))
    SLCWVds['CWAVE_20_SLC'] = xarray.DataArray(spectralmoments.T, dims=['time_sar', 'cwave_coords'],
                                                   coords={'time_sar': SLCWVds['time_sar'].values,
                                                           'cwave_coords': np.arange(20)},
                                                   attrs={
                                                       'description': '20 CWAVE params computed on SLC cartesian cross spectra prepared using xrft/xsar'})

    # to be continued here 2nd June 2022....
    # if grid_xspec == 'cartesian':
    max_KI = 2 * np.pi / 50.
    logging.info('max_KI %s', max_KI)
    SLCWVds['cartRexspec'] = z_re_new_da
    SLCWVds['cartImxspec'] = z_im_new_da
    # SLCWVds = store_cart_xspec_Ntau(
    #     z_re_new_da.where((abs(z_re_new_da.kx) < max_KI) & (abs(z_re_new_da.ky) < max_KI), drop=True),
    #     z_im_new_da.where((abs(z_re_new_da.kx) < max_KI) & (abs(z_re_new_da.ky) < max_KI), drop=True)
    #     , SLCWVds)
    # elif grid_xspec == 'polar':
    # xsspecCross_Polar_Re_fullspan, xsspecCross_Polar_Im_fullspan = store_pol_xspec_Ntau(all_computed_cart_xspectrum,
    #                                                                                     nperseg=nperseg,
    #                                                                                     platform_heading=ta)
    # z_re_tmp_no_interpolation = np.abs(all_computed_cart_xspectrum['cross-spectrum_2tau'].mean(dim='2tau').real)
    # coV, azimuuth_cutoff = get_cut_off_profile(z_re_tmp_no_interpolation, display_cutoff=False)
    SLCWVds['azimuth_cutoff'] = xarray.DataArray([all_computed_cart_xspectrum['cutoff'].values], dims=['time_sar'],
                                                 attrs=all_computed_cart_xspectrum['cutoff'].attrs)
    # SLCWVds['crossSpectraRePol'] = xsspecCross_Polar_Re_fullspan
    # SLCWVds['crossSpectraRePol'].data = xsspecCross_Polar_Re_fullspan.values
    # SLCWVds['crossSpectraImPol'] = xsspecCross_Polar_Im_fullspan
    # SLCWVds['crossSpectraImPol'].data = xsspecCross_Polar_Im_fullspan.values
    return SLCWVds

def add_xspec_subdomains(subsetcoloc,allspecs_per_sub_domain,grid_xspec,nperseg,ta,splitting_image,lons_all,lats_all):
    """

    :param subsetcoloc:
    :param allspecs_per_sub_domain:
    :param grid_xspec:
    :param nperseg:
    :param ta:
    :param splitting_image:
    :param lons_all:
    :param lats_all:
    :return:
    """
    Re_subs = None
    Im_subs = None
    for rect_id in allspecs_per_sub_domain:
        if grid_xspec == 'cartesian':
            subsetcoloc = store_cart_xspec_Ntau(allspecs_per_sub_domain[rect_id], subsetcoloc, nperseg,
                                                prefix=str(rect_id) + '-')
        elif grid_xspec == 'polar':
            xsspecCross_Polar_Re_subtmp, xsspecCross_Polar_Im_subtmp = store_pol_xspec_Ntau(
                allspecs_per_sub_domain[rect_id],
                nperseg=nperseg, platform_heading=ta, prefix='dom' + str(rect_id) + '-')
            if Re_subs is None:
                Re_subs = xsspecCross_Polar_Re_subtmp
                Im_subs = xsspecCross_Polar_Im_subtmp
            else:
                Re_subs = xarray.concat([Re_subs, xsspecCross_Polar_Re_subtmp], dim='sub_domain')
                Im_subs = xarray.concat([Im_subs, xsspecCross_Polar_Im_subtmp], dim='sub_domain')
        else:
            raise Exception('unknown grid x spec %s' % grid_xspec)
    subsetcoloc['xspec_polar_Re_sub_domains'] = Re_subs
    subsetcoloc['xspec_polar_Im_sub_domains'] = Im_subs
    subsetcoloc.assign_coords({'sub_domain': np.arange(len(allspecs_per_sub_domain))})
    # add geolocations of the subdomains
    # store long/lat of each rectangles
    geoloc = {}
    for rect_x in splitting_image:
        sli_az, sli_ra = splitting_image[rect_x]['azimuth'], splitting_image[rect_x]['range']
        lons = [
            lons_all[sli_az.start, sli_ra.start],
            lons_all[sli_az.start, sli_ra.stop],
            lons_all[sli_az.stop, sli_ra.stop],
            lons_all[sli_az.stop, sli_ra.start],
            lons_all[sli_az.start, sli_ra.start],
        ]
        lats = [
            lats_all[sli_az.start, sli_ra.start],
            lats_all[sli_az.start, sli_ra.stop],
            lats_all[sli_az.stop, sli_ra.stop],
            lats_all[sli_az.stop, sli_ra.start],
            lats_all[sli_az.start, sli_ra.start],
        ]
        if 'lons' not in geoloc:
            geoloc = {'lons': np.array(lons), 'lats': np.array(lats)}
        else:
            geoloc['lons'] = np.vstack([geoloc['lons'], np.array(lons)])
            geoloc['lats'] = np.vstack([geoloc['lats'], np.array(lats)])
    logging.info('rects lons : %s', geoloc['lons'].shape)
    subsetcoloc['lons_rects'] = xarray.DataArray(geoloc['lons'], dims=['nb_domains', 'dim_poly'],
                                                 coords={'dim_poly': np.arange(5),
                                                         'nb_domains': np.arange(len(splitting_image))})
    subsetcoloc['lats_rects'] = xarray.DataArray(geoloc['lats'], dims=['nb_domains', 'dim_poly'],
                                                 coords={'dim_poly': np.arange(5),
                                                         'nb_domains': np.arange(len(splitting_image))})
    return subsetcoloc


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
    all_subsets_coloc = []
    ds_coloc = read_coloc_file(sar_unit,date_day,alti_mission)
    if ds_coloc:
        list_tiff_paths = get_tiff_path(ds_coloc,sar_unit)
        logging.info('Nb tiff path: %s while expected :%s',len(list_tiff_paths),ds_coloc['lon_SAR'].size)
        list_base_tiff = [os.path.basename(hh) for hh in list_tiff_paths]
        ds_coloc['tiff'] = xarray.DataArray(np.array(list_base_tiff),dims=['time_sar']) #,coords={'time_sar':ds_coloc['time_sar']}

        nb_match = len(list_tiff_paths)

        #for xxx,indxs in enumerate(inds):
        for xxx in range(nb_match):
            logging.info('prepare coloc : %s/%s', xxx + 1, nb_match)
            logging.info('peak memory usage: %s Mbytes', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
            one_tiff = list_tiff_paths[xxx]
            SLCWVds = get_SAR_SLC_quantities(one_tiff)
            # 1) read all params from Justin s dataset
            subsetcoloc0 = ds_coloc.isel(
                time_sar=np.array([xxx]))  # { on peut selectionner plusieurs indice en meme temps avec isel}
            logging.debug('subsetcoloc : %s', subsetcoloc0)
            subsetcoloc0 = subsetcoloc0.rename({'lon_SAR': 'lonSAR', 'lat_SAR': 'latSAR'})
            subsetcoloc0['swh'].attrs[
                'description'] = 'SWH predicted by Quach et al 2020 NN algorithm and annotated in CCI sea state orbit files'
            subsetcoloc0['hs_alti_mean'].attrs[
                'description'] = 'average SWH given by CMEMS WAV colocated altimeters'
            subsetcoloc0['hs_alti_std'].attrs[
                'description'] = 'standard deviation of SWH given by CMEMS WAV colocated altimeters'
            subsetcoloc0['hs_alti_count'].attrs[
                'description'] = 'number of altimeters LRM L2 CMEMS WAV points (1Hz) used to compute hs_alti_mean and hs_alti_std'
            subsetcoloc = xarray.merge([subsetcoloc0,SLCWVds])
            # else:
            #     raise Exception('unknown grid x spec %s'%grid_xspec)

            #add spectrum per sub domain
            add_xspec_subdoms = False
            if add_xspec_subdoms:
                add_xspec_subdomains(subsetcoloc, allspecs_per_sub_domain, grid_xspec, nperseg, ta, splitting_image,
                                     lons_all, lats_all)
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
    glob_attrs = {'step1_processing_method': save_training_file.__name__,
                  'step1_processing_script': os.path.basename(__file__),
                  'step1_processing_env': sys.executable,
                  'step1_processing_date': datetime.datetime.today().strftime('%Y%m%d %H:%M'),
                  'step1_input_dir': 'IFREMER S1 WV SLC data + ' + DIR_ORIGINAL_COLOCS,
                  'step1_outputdir_dir': os.path.dirname(outputfile)
                  }
    dscoloc_enriched.attrs = glob_attrs
    dscoloc_enriched.attrs['created_on'] = '%s' % datetime.datetime.today()
    dscoloc_enriched.attrs['created_by'] = 'Antoine Grouazel'
    dscoloc_enriched.attrs['purpose'] = 'SAR Hs NN regression'
    dscoloc_enriched.attrs['content'] = ' SAR + Alti + ww3 colocations'
    for vv in dscoloc_enriched:
        logging.debug('%s attrs: %s',vv,dscoloc_enriched[vv].attrs)
    dscoloc_enriched = dscoloc_enriched.drop('spatial_ref')
    dscoloc_enriched = dscoloc_enriched.drop('pol')
    t_save = time.time()
    dscoloc_enriched.to_netcdf(outputfile)
    logging.info('outputfile : %s',outputfile)
    os.chmod(outputfile,0o0777)
    logging.info('set permission 777 on output file done')
    logging.info('time ot save file %1.1f seconds',(time.time()-t_save))


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
    parser.add_argument('--alti',choices=POSSIBLES_CMEMS_ALTI.keys(),required=True,help='alti mission in %s'%(POSSIBLES_CMEMS_ALTI.keys()))
    parser.add_argument('--outputdir',action='store',default=OUTPUTDIR,required=False,help='outputdir [optional, default is %s]'%OUTPUTDIR)
    args = parser.parse_args()
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    t1 = time.time()
    #time.sleep(np.random.randint(0,5,1)[0]) # to avoid mkdir issues with p-run
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
                              datedt.strftime('%j'),'training_D_jan23_%s_%s_%s.nc' %(args.date,args.alti,args.sar_unit))
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
        list_dscoloc_enriched = read_all_SAR_variables(sar_unit=args.sar_unit,alti_mission=POSSIBLES_CMEMS_ALTI[args.alti],
                                                       date_day=date_to_treat_dt,dev=args.dev,grid_xspec='polar')
        if len(list_dscoloc_enriched)>0:
            #dscoloc_enriched = xarray.merge(list_dscoloc_enriched) # here merge and concat are doing bullshit (increase kx and ky size...) -> I do it by myself
            #before concat I need to align number of subdomains (on the minimum nber of pixels avail)
            add_subdomains = False
            if add_subdomains:
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
                if len(new_list_ds)>0:
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
            else:
                save_training_file(xarray.concat(list_dscoloc_enriched,dim='time_sar'),outputfile)
    logging.info('analysis done in %s seconds',time.time()-t1)
    logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
