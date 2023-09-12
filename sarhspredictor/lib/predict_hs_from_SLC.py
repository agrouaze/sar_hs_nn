"""
A Grouazel
7 June 2021
I need a script to store functions shared for building the training dataset and to perform inferences with NN model to validate the training
"""
import pdb

import os
import numpy as np
import logging
import glob
import xsarsea
import time
import copy
import netCDF4
import datetime
import xarray
import xsar
import spectrum_clockwise_to_trigo
import spectrum_rotation
import conversion_polar_cartesian #from xsarseafork
import xsarsea.cross_spectra_core
#import cross_spectra_core_v2 #version dans fork agrouaze May 2022
from sarhspredictor.lib.compute_CWAVE_params import format_input_CWAVE_vector_from_OCN
from sarhspredictor.lib.sarhs import preprocess
from sarhspredictor.lib.predict_with_quach2020_on_OCN_using_keras import main_level_0
import match_L1_L2_measurement #mpc S1 data_collect
reference_oswK_1145m_60pts = np.array([0.005235988,0.00557381,0.005933429,0.00631625,0.00672377,
                                       0.007157583,0.007619386,0.008110984,0.008634299,0.009191379,
                                       0.0097844,0.01041568,0.0110877,0.01180307,0.01256459,0.01337525,
                                       0.01423822,0.01515686,0.01613477,0.01717577,0.01828394,0.01946361,
                                       0.02071939,0.02205619,0.02347924,0.02499411,0.02660671,0.02832336,
                                       0.03015076,0.03209607,0.03416689,0.03637131,0.03871796,0.04121602,
                                       0.04387525,0.04670605,0.0497195,0.05292737,0.05634221,0.05997737,
                                       0.06384707,0.06796645,0.0723516,0.07701967,0.08198893,0.08727881,
                                       0.09290998,0.09890447,0.1052857,0.1120787,0.1193099,0.1270077,
                                       0.1352022,0.1439253,0.1532113,0.1630964,0.1736193,0.1848211,
                                       0.1967456,0.2094395])


def from_np64_to_dt(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)


def get_xspectrum_SLC(slc,nb_match,dev=False,resolution=None,resampling=None):
    """

    :param slc:
    :param nb_match:
    :param dev:
    :params resampling : rasterio.enums.Resampling.rms for instance
    :return:
    """
    datedt_slc = datetime.datetime.strptime(os.path.basename(slc).split('-')[4],
                                            '%Y%m%dt%H%M%S')  # cette date change bien mem pour les data en 2015Z
    #sub_swath_IDs = xsar.Sentinel1Meta(os.path.dirname(os.path.dirname(slc))).subdatasets
    imagette_number = os.path.basename(slc).split('-')[-1].replace('.tiff', '')
    # jouvre le smeta pour ensuite verifier que lindice trouver par ma methode es ok
    wv_slc_meta = xsar.sentinel1_meta.Sentinel1Meta(
        "SENTINEL1_DS:%s:WV%s" % (os.path.dirname(os.path.dirname(slc)), imagette_number))
    #indice = xsarsea.cross_spectra_core_v2.get_imagette_indice(slc, wv_slc_meta)
    #print(indice,type(indice),imagette_number,type(imagette_number))
    #assert str(indice)==imagette_number
    # jouvre le dataset
    dsslc_raw = xsar.sentinel1_dataset.Sentinel1Dataset("SENTINEL1_DS:%s:WV_%s" % (os.path.dirname(os.path.dirname(slc)),imagette_number))
    dsslc = xsarsea.cross_spectra_core.read_slc(dsslc_raw.dataset)
    # if resolution is not None:
    #     if resolution['atrack'] == 1 and resolution['xtrack'] == 1:
    #         # June 2021, a patch because currently resolution 1:1 for image and rasterio returns an error
    #         s1ds = xsar.Sentinel1Dataset(wv_slc_meta, resolution=None, resampling=None)
    #     else:
    #         s1ds = xsar.Sentinel1Dataset(wv_slc_meta, resolution=resolution, resampling=resampling)
    # else:
    #     s1ds = xsar.Sentinel1Dataset(wv_slc_meta)
    #je fais qlq changement mineur sur les coords
    #dsslc = cross_spectra_core_v2.read_slc(dsslc_raw)

    if dev:
        nperseg = {'range' : 2048,'azimuth' : 2048}
    else:
        nperseg = {'range' : 512,'azimuth' : 512}
    t0 = time.time()
    #allspecs,frange,fazimuth,allspecs_per_sub_domain,splitting_image, \
    #limits_sub_images \
    allspecs = xsarsea.cross_spectra_core.compute_SAR_cross_spectrum2(dsslc['digital_number'].isel(pol=0), N_look=3,
                                                                      look_width=0.25, look_overlap=0., look_window=None,
                                range_spacing=None, welsh_window='hanning', nperseg=nperseg,
                                noverlap={'range': 256, 'azimuth': 256}, spacing_tol=1e-3)
    # allspecs, tfazi, tfran    = cross_spectra_core_v2.compute_SAR_cross_spectrum2(
    #     dsslc['digital_number'].isel(pol=0),
    #     N_look=3,look_width=0.25,
    #     look_overlap=0.,look_window=None,  # range_spacing=slc.attrs['rangeSpacing']
    #     welsh_window='hanning',
    #     nperseg=nperseg,
    #     noverlap={'range' : 256,'azimuth' : 256}
    #     ,spacing_tol=1e-3,debug_plot=False,return_periodoXspec=False)
    logging.info('time to get %s X-spectra : %1.1f seconds',len(allspecs),time.time() - t0)
    # 3) interpolate and convert cartesian grid to polar 72,60
    xspecRe = np.abs(allspecs['cross-spectrum_2tau'].mean(dim='2tau').real)
    # if False:
    #     crossSpectraRePol = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(xspecRe,Nphi=72,
    #                                                                                ksampling='log',**{'Nk' : 60,'kmin' :
    #         reference_oswK_1145m_60pts[0],'kmax' : reference_oswK_1145m_60pts[-1]})
    # else:# version corrected January 2022
    crossSpectraRePol = conversion_polar_cartesian.from_xCartesianSpectrum(xspecRe,Nphi=72,
                                                                                ksampling='log',
                                                                                **{'k' : reference_oswK_1145m_60pts})
    crossSpectraRePol = spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(
        crossSpectraRePol)
    #crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol,90.)  # This is for having origin at North
    crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol, -90.)  # This is for having origin at North, correction January 2022
    crossSpectraRePol = spectrum_rotation.apply_rotation(crossSpectraRePol,dsslc.attrs['platform_heading'])
    crossSpectraRePolval = copy.copy(crossSpectraRePol.values.squeeze())

    xspecIm = allspecs['cross-spectrum_2tau'].mean(dim='2tau').imag # correction 19 May 2022,  removed the abs()
    # if False:
    #     crossSpectraImPol = xsarsea.conversion_polar_cartesian.from_xCartesianSpectrum(xspecIm,Nphi=72,
    #                                                                                ksampling='log',**{'Nk' : 60,'kmin' :
    #         reference_oswK_1145m_60pts[0],'kmax' : reference_oswK_1145m_60pts[-1]})
    # else:  # version corrected January 2022
    crossSpectraImPol = conversion_polar_cartesian.from_xCartesianSpectrum(xspecIm, Nphi=72,
                                                                                   ksampling='log',
                                                                                   **{
                                                                                       'k': reference_oswK_1145m_60pts})
    crossSpectraImPol = spectrum_clockwise_to_trigo.apply_clockwise_to_trigo(
        crossSpectraImPol)
    crossSpectraImPol = spectrum_rotation.apply_rotation(crossSpectraImPol,-90.)  # This is for having origin at North, -90 instead of 90 January 22
    crossSpectraImPol = spectrum_rotation.apply_rotation(crossSpectraImPol,dsslc.attrs['platform_heading'])
    crossSpectraImPolval = copy.copy(crossSpectraImPol.values.squeeze())

    # duplicate the X spectra and CWAVE params for the N number of colcoations with this particular SAR acquisition
    #multi_xpsec_im = np.tile(crossSpectraImPol.values,(nb_match,1,1))
    #multi_xpsec_re = np.tile(crossSpectraRePol.values,(nb_match,1,1))
    multi_xpsec_im = crossSpectraImPol.values[np.newaxis,:,:]
    multi_xpsec_re = crossSpectraRePol.values[np.newaxis,:,:]
    logging.info('multi_xpsec_im %s',multi_xpsec_im.shape)
    times_bidons = [
        datedt_slc]  # dirty trick, j invente des dates pour pouvoir ensuite aggregger les colocs ensemble a coup de mfdataset
    # for ttt in range(1,nb_match) :
    #     times_bidons.append(datedt_slc + datetime.timedelta(seconds=ttt))
    logging.info('times_bidons : %s',times_bidons)
    crossSpectraImPol_xa = xarray.DataArray(multi_xpsec_im,dims=['time','k','phi'],
                                            coords={'time' : times_bidons,
                                                    'k' : crossSpectraImPol.k.values,
                                                    'phi' : crossSpectraImPol.phi.values
                                                    })
    crossSpectraRePol_xa = xarray.DataArray(multi_xpsec_re,dims=['time','k','phi'],
                                            coords={'time' : times_bidons,
                                                    'k' : crossSpectraRePol.k.values,
                                                    'phi' : crossSpectraRePol.phi.values
                                                    })
    return crossSpectraImPol_xa,crossSpectraRePol_xa,crossSpectraImPolval,crossSpectraRePolval,datedt_slc,times_bidons


def compute_Cwave_params_and_xspectra_fromSLC(slc,dev,ths1,ta,s0,nv,lonsar,latsar,incidenceangle,nb_match=1):
    """

    :param slc: str
    :param dev: bool True-> larger periodograms to speed up the computation of x spectra
    :param ths1: np.array degrees
    :param ta: track angle in degree
    :param s0: float NRCS |dB]
    :param nv: float normalized variance such as oswNv variable
    :param lonsar: one float degree
    :param latsar: one float degree
    :param incidenceangle: one float degree
    :param nb_match: integer to replicate the Xspectra and S params as much as we could have colocated altimeter points (default is 1)
    :return:
    """
    # 2) read X spectra from tiff


    sat = os.path.basename(slc)[0:3]

    crossSpectraImPol_xa,crossSpectraRePol_xa,crossSpectraImPolval,crossSpectraRePolval,datedt_slc,times_bidons = \
        get_xspectrum_SLC(slc,nb_match=nb_match,dev=dev)
    #subsetcoloc['crossSpectraImPol'] = crossSpectraImPol # version simple si il ny avait pas eu des multi colocs
    #subsetcoloc['crossSpectraRePol'] = crossSpectraRePol
    subset_ok,flagKcorrupted,cspcReX,cspcImX,cspcRev2,ks1,ths1,kx,ky,cspcReX_not_conservativ,S = format_input_CWAVE_vector_from_OCN(
        cspcRe=crossSpectraRePolval,cspcIm=crossSpectraImPolval,ths1=ths1,ta=ta,
        incidenceangle=incidenceangle,s0=s0,nv=nv,ks1=reference_oswK_1145m_60pts,
        datedt=datedt_slc,lonSAR=lonsar,latSAR=latsar,satellite=sat)
    logging.info('S size: %s',S.shape)
    assert np.isfinite(S).all()
    return crossSpectraImPol_xa,crossSpectraRePol_xa,times_bidons,S


def prepare_prediction_hs_computing_xspectra(slc,dev=False):
    ocn = match_L1_L2_measurement.getNCcorresponding2TIFF(slc)
    ds_ocn = xarray.open_dataset(ocn)
    ths1 = np.arange(0,360,5)
    ta= ds_ocn['oswHeading'].values[0][0]
    s0 = ds_ocn['oswNrcs'].values[0][0]
    nv = ds_ocn['oswNv'].values[0][0]
    lonsar = ds_ocn['oswLon'].values[0][0]
    latsar = ds_ocn['oswLat'].values[0][0]
    incidenceangle = ds_ocn['oswIncidenceAngle'].values[0][0]
    oswWindSpeed = ds_ocn['oswWindSpeed'].values[0][0]
    oswLandFlag = ds_ocn['oswLandFlag'].values[0][0]
    crossSpectraImPol_xa,crossSpectraRePol_xa,times_bidons,S = compute_Cwave_params_and_xspectra_fromSLC(slc,
                                                                                                         dev,
                                                                                                         nb_match=1,
                                                                                                         ths1=ths1,
                                                                                                         ta=ta,
                                                                                                         s0=s0,
                                                                                                         nv=nv,
                                                                                                         incidenceangle=incidenceangle,
                                                                                                         lonsar=lonsar,
                                                                                                         latsar=latsar)
    return crossSpectraImPol_xa,crossSpectraRePol_xa,S,s0,nv,lonsar,latsar,incidenceangle,ths1,ta,oswWindSpeed,oswLandFlag

def predictions_from_slc_and_ocn(slc,s0,nv,ta,ths1,lonsar,latsar,oswWindSpeed,oswLandFlag,incidenceangle,crossSpectraImPol_xa,crossSpectraRePol_xa,S,model,dev=False):
    """
    note: I decide to read some informations from OCN associated file to be as close as possible to the content of cwaveV4
     that was used as a basis in the training dataset of exp1 (the one using the Xspectra from SLC + xrft)
    :param slc: str full path of WV tiff
    :param
    :return:
    """
    ds_4_future_inferences = xarray.Dataset()
    datedt_slc = [datetime.datetime.strptime(os.path.basename(slc).split('-')[4],
                                            '%Y%m%dt%H%M%S')]
    timeSAR_vals  = datedt_slc #je nai qu une valeur ici pcq je traite les SLC une par une


    sat = os.path.basename(slc)[0 :3]
    sattelites = [sat]
    satellites_int = np.array([threelettersat[2] == 'a' for threelettersat in sattelites]).astype(int)

    # cette partie ressemble fortement a prepare_training_dataset_core() mais il y a des differences sur le dataset en input
    # il ne faut pas quil y ait de diff en sortie comme je pourrais reutiliser les methodes pour la predictions
    #ds_4_future_inferences['py_S'] = S

    varstoadd = ['S','cwave','dxdt','latlonSARcossin','todSAR','oswIncidenceAngle','oswWindSpeed','oswLandFlag','heading',
                 'incidence','incidence_angle','satellite','py_cspcImX','py_cspcReX','oswLon','oswLat','nv','nrcs']
    #additional vars that could help for the validation
    varstoadd += ['fileNameFull']
    for vv in varstoadd :
        logging.info('start format variable :%s',vv)
        if vv in ['cwave'] :
            dimszi = ['time','cwavedim']
            coordi = {'time' : datedt_slc,'cwavedim' : np.arange(22)}
            logging.debug('S %s s0: %s nv: %s',S.shape,s0.shape,nv.shape)
            cwave = np.vstack([S,np.array([s0]),np.array([nv])]).T  # found L77 in preprocess.py
            logging.debug('cwave vals: %s',cwave.shape)
            cwave = preprocess.conv_cwave(cwave)
            logging.debug('cwave shape : %s',cwave.shape)
            ds_4_future_inferences[vv] = xarray.DataArray(data=cwave,dims=dimszi,coords=coordi)
        elif vv in ['fileNameFull','fileNameL2']:
            # dimszi = ['time','pathnchar']
            # coordi = {'time' : timeSAR_seconds,'pathnchar' : len(fpaths[0])}
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            ds_4_future_inferences[vv] = xarray.DataArray(data=[slc],dims=dimszi,coords=coordi)
        elif vv in ['oswLon']:
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            ds_4_future_inferences[vv] = xarray.DataArray(data=[lonsar],dims=dimszi,coords=coordi)
        elif vv in ['oswLat']:
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            ds_4_future_inferences[vv] = xarray.DataArray(data=[latsar],dims=dimszi,coords=coordi)
        elif vv == 'S' :  # to ease the comparison with Justin files
            dimszi = ['time','Sdim']
            coordi = {'time' : datedt_slc,'Sdim' : np.arange(20)}
            ds_4_future_inferences[vv] = xarray.DataArray(data=S.T,dims=dimszi,coords=coordi)
        elif vv in ['dxdt'] :  # dx and dt and delta from coloc with alti see /home/cercache/users/jstopa/sar/empHs/cwaveV5, I can put zeros here at this stage
            #dxdt = np.column_stack([ds_train_raw['dx'].values,ds_train_raw['dt'].values])
            dxdt = np.column_stack([np.zeros(s0.shape),np.ones(s0.shape)])
            dimszi = ['time','dxdtdim']
            coordi = {'time' : datedt_slc,'dxdtdim' : np.arange(2)}
            ds_4_future_inferences[vv] = xarray.DataArray(data=dxdt,dims=dimszi,coords=coordi)
        elif vv in ['latlonSARcossin'] :
            latSARcossin = preprocess.conv_position(latsar)  # Gets cos and sin
            lonSARcossin = preprocess.conv_position(lonsar)
            latlonSARcossin = np.hstack([latSARcossin,lonSARcossin])
            dimszi = ['time','latlondim']
            coordi = {'time' : datedt_slc,'latlondim' : np.arange(4)}
            ds_4_future_inferences[vv] = xarray.DataArray(data=latlonSARcossin,dims=dimszi,coords=coordi)
        elif vv in ['todSAR'] :
            dimszi = ['time']
            logging.debug('timeSAR_vals : %s',timeSAR_vals)
            #new_dates_dt = np.array([from_np64_to_dt(dt64) for dt64 in timeSAR_vals])
            unit = "hours since 2010-01-01T00:00:00Z UTC"  # see https://github.com/grouny/sar_hs_nn/blob/c05322e6635c6d77409e36537d7c3b58788e7322/sarhspredictor/lib/sarhs/preprocess.py#L11
            new_dates_num = np.array([netCDF4.date2num(dfg,unit) for dfg in timeSAR_vals])
            coordi = {'time' : datedt_slc}
            todSAR = preprocess.conv_time(new_dates_num)
            ds_4_future_inferences[vv] = xarray.DataArray(data=todSAR,dims=dimszi,coords=coordi)
        elif vv in ['oswK'] :
            dimszi = ['time','oswWavenumberBinSize']
            coordi = {'time' : datedt_slc,'oswWavenumberBinSize' : np.arange(len(reference_oswK_1145m_60pts))}
            ds_4_future_inferences[vv] = xarray.DataArray(data=reference_oswK_1145m_60pts,dims=dimszi,coords=coordi)
        elif vv in ['incidence',] :
            dimszi = ['time','incdim']
            coordi = {'time' : datedt_slc,'incdim' : np.arange(2)}
            incidenceangle_tmp = np.array([incidenceangle])
            logging.debug('incidenceangle_tmp %s %s',type(incidenceangle_tmp),incidenceangle_tmp)
            incidenceangle_conv = preprocess.conv_incidence(incidenceangle_tmp)
            ds_4_future_inferences[vv] = xarray.DataArray(data=incidenceangle_conv,dims=dimszi,coords=coordi)
        elif vv == 'oswLandFlag':
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            oswLandFlag = np.array([oswLandFlag])
            ds_4_future_inferences[vv] = xarray.DataArray(data=oswLandFlag,dims=dimszi,coords=coordi)
        elif vv == 'oswWindSpeed':
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            oswWindSpeed = np.array([oswWindSpeed])
            ds_4_future_inferences[vv] = xarray.DataArray(data=oswWindSpeed,dims=dimszi,coords=coordi)
        elif vv in ['incidence_angle'] :
            dimszi = ['time']
            coordi = {}
            coordi['time'] = datedt_slc
            incidence_angle_val = np.array([incidenceangle])
            logging.debug('incidence_angle_val %s %s',type(incidence_angle_val),incidence_angle_val)
            ds_4_future_inferences[vv] = xarray.DataArray(data=incidence_angle_val,dims=dimszi,coords=coordi)
        elif vv == 'oswIncidenceAngle':
            dimszi = ['time']
            coordi = {}
            coordi['time'] = datedt_slc
            osw_incidence = np.array([incidenceangle])
            logging.debug('osw_incidence %s %s',type(osw_incidence),osw_incidence)
            ds_4_future_inferences[vv] = xarray.DataArray(data=osw_incidence,dims=dimszi,coords=coordi)
        elif vv in ['satellite'] :
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            # satellite_int = np.array([satellite[2] == 'a']).astype(int)
            ds_4_future_inferences[vv] = xarray.DataArray(data=satellites_int,dims=dimszi,coords=coordi)
        elif vv in ['platformName'] :
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            satellite_int = sattelites
            ds_4_future_inferences[vv] = xarray.DataArray(data=satellite_int,dims=dimszi,coords=coordi)
        elif vv in ['nrcs'] :
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            ds_4_future_inferences[vv] = xarray.DataArray(data=s0,dims=dimszi,coords=coordi)
        elif vv in ['heading'] :
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            ds_4_future_inferences[vv] = xarray.DataArray(data=[ta],dims=dimszi,
                                                          coords=coordi)
            logging.debug('heading added')
        elif vv in ['nv'] :
            dimszi = ['time']
            coordi = {'time' : datedt_slc}
            ds_4_future_inferences[vv] = xarray.DataArray(data=nv,dims=dimszi,coords=coordi)
        # elif vv in ['oswQualityCrossSpectraRe','oswQualityCrossSpectraIm'] :
        #     if vv == 'oswQualityCrossSpectraRe' :
        #         datatmp = cspcRe
        #     elif vv == 'oswQualityCrossSpectraIm' :
        #         datatmp = cspcIm
        #     else :
        #         raise Exception()
        #     # datatmp = ds[vv].values.squeeze()
        #     # olddims = [x for x in ds[vv].dims if x not in ['oswAzSize','oswRaSize']]
        #     coordi = {}
        #     # for didi in olddims:
        #     #    coordi[didi] = ds[vv].coords[didi].values
        #     coordi['time'] = datedt_slc
        #     coordi['oswAngularBinSize'] = np.arange(len(ths1))
        #     coordi['oswWavenumberBinSize'] = np.arange(len(reference_oswK_1145m_60pts))
        #     dimsadd = ['time','oswAngularBinSize','oswWavenumberBinSize']
        #     # if datatmp.shape == (72,60) :  # case only one spectra
        #     #    datatmp = datatmp.reshape((1,72,60))
        #
        #     ds_4_future_inferences[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        elif vv =='py_cspcImX': #here xspectra from xrft SLC
            ds_4_future_inferences['crossSpectraImPol'] = crossSpectraImPol_xa
        elif vv == 'py_cspcReX' :  # here xspectra from xrft SLC
            ds_4_future_inferences['crossSpectraRePol'] = crossSpectraRePol_xa
        # elif vv in ['py_cspcImX','py_cspcReX'] :
        #     datatmp = ds_train_raw[vv].values
        #     coordi = ds_train_raw[vv].coords
        #     coordi['time'] = datedt_slc
        #     dimsadd = ds_train_raw[vv].dims
        #     ds_4_future_inferences[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        else :
            raise Exception('not handle variable: %s'%vv)
            # datatmp = ds_train_raw[vv].values.squeeze()
            # olddims = [x for x in ds_train_raw[vv].dims if x not in ['oswAzSize','oswRaSize']]
            # coordi = {}
            # for didi in olddims :
            #     coordi[didi] = ds_train_raw[vv].coords[didi].values
            # coordi['time'] = datedt_slc
            # dimsadd = ['time']
            # logging.info('data: %s',datatmp.shape)
            # ds_4_future_inferences[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        # logging.debug('field xarray : %s %s',vv,newds[vv])
    logging.debug('newds: %s',ds_4_future_inferences)
    logging.info('SAR data ready to be used')
    # cspcRe = ds_train_raw['oswQualityCrossSpectraRe'].values
    # cspcIm = ds_train_raw['oswQualityCrossSpectraIm'].values
    logging.debug('crossSpectraRePol_xa : %s',crossSpectraRePol_xa.shape)
    logging.info('re before norm: %s min : %1.3f max: %1.3f',crossSpectraRePol_xa.values.shape,crossSpectraRePol_xa.values.min(),crossSpectraRePol_xa.values.max())
    logging.info('im before norm: %s min : %1.3f max: %1.3f',crossSpectraImPol_xa.values.shape,crossSpectraImPol_xa.values.min(),crossSpectraImPol_xa.values.max())
    re = preprocess.conv_real(np.swapaxes(crossSpectraRePol_xa.values,1,2),exp_id=1)
    im = preprocess.conv_imaginary(np.swapaxes(crossSpectraImPol_xa.values,1,2),exp_id=1)
    logging.info('re : %s min : %1.3f max: %1.3f',re.shape,re.min(),re.max())
    logging.info('im : %s min : %1.3f max: %1.3f',im.shape,im.min(),re.max())
    spectrum = np.stack((re,im),axis=3)
    logging.info('spectrum shape : %s',spectrum.shape)
    ds_4_future_inferences = main_level_0(ds_4_future_inferences,spectrum,model)
    logging.info('safe ds with Hs : %s',ds_4_future_inferences)
    return spectrum,ds_4_future_inferences


def predictions_from_slc_SAFE(safe_path,model,dev=False):
    """
    do Hs predictions on a full SAFE WV SLC
    :param safe_path:
    :return:
    """
    list_tiff = sorted(glob.glob(os.path.join(safe_path,'measurement','*.tiff')))
    all_ds = []
    spectrum = None
    for tti,ff in enumerate(list_tiff):
        logging.info('%s/%s %s',tti,len(list_tiff),os.path.basename(ff))
        crossSpectraImPol_xa,crossSpectraRePol_xa,S,s0,nv,lonsar,latsar,incidenceangle,ths1,ta,oswWindSpeed,oswLandFlag\
            = prepare_prediction_hs_computing_xspectra(ff,dev=dev)
        spectrum,tmpds = predictions_from_slc_and_ocn(ff,s0,nv,ta,ths1,lonsar,latsar,oswWindSpeed,oswLandFlag,incidenceangle,
                                                      crossSpectraImPol_xa,crossSpectraRePol_xa,S,model=model,dev=dev)
        tmpds = tmpds.drop('k') # drop the k for concatenation (dont know if it is a bug in xarray concat...
        all_ds.append(tmpds)
        if dev and tti==1:
            logging.info('break for dev test')
            break
    logging.info('concatenation of all the individual prediction performed on tiff o have a single dataset for the SAFE')
    safe_ds = xarray.concat(all_ds,dim='time')
    logging.info('safe_ds : %s',safe_ds)
    return safe_ds


