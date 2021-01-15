
"""

"""
import logging
import xarray
import os
import datetime
import sys
import time
import numpy as np
sys.path.append('/home1/datahome/agrouaze/git/SAR-Wave-Height/')
from sarhs import preprocess
sys.path.append('/home1/datahome/agrouaze/git/mpc/qualitycheck/')
sys.path.append('/home1/datahome/agrouaze/git/npCWAVE/')
import compute_hs_total_SAR_v2


def preproc_ocn_wv(ds):
    """

    :param ds:
    :return:
    """
    filee = ds.encoding["source"]
    logging.debug('filee %s',os.path.basename(filee))
    fdatedt = datetime.datetime.strptime(os.path.basename(filee).split('-')[4],'%Y%m%dt%H%M%S')
    logging.debug('fdatedt : %s %s',fdatedt,type(fdatedt))
    #ds['time'] = xarray.DataArray([fdatedt],dims=['time']) # marche avec derniere version de xarray pas ancienne
    logging.debug('brut ds: %s',ds)
    try:
        ds['time'] = xarray.DataArray(np.array([fdatedt]),dims=['time'],coords={'time':[0]})
        ds = ds.sortby('time',ascending=True)
    except:
        pass
    newds = xarray.Dataset()

    #format data for CWAVE 22 params computation
    cspcRe = ds['oswQualityCrossSpectraRe'].values.squeeze().T
    cspcIm = ds['oswQualityCrossSpectraIm'].values.squeeze().T
    ths1 = ds['oswPhi'].values.squeeze()
    ks1 = ds['oswK'].values.squeeze()
    ta = ds['oswHeading'].values.squeeze()
    incidenceangle =ds['oswIncidenceAngle'].values.squeeze()
    s0 =  ds['oswNrcs'].values.squeeze()
    nv = ds['oswNv'].values.squeeze()
    lonSAR = ds['oswLon'].values.squeeze()
    latSAR = ds['oswLat'].values.squeeze()
    satellite = os.path.basename(filee)[0:3]
    subset_ok,flagKcorrupted,cspcReX,cspcImX,cspcRe,ks1,ths1,kx,ky,\
    cspcReX_not_conservativ,S = compute_hs_total_SAR_v2.format_input_CWAVE_vector_from_OCN(cspcRe,
                                                                            cspcIm,ths1,ta,incidenceangle,
                                                                            s0,nv,ks1,fdatedt,lonSAR,latSAR,satellite)

    varstoadd = ['S','cwave', 'dxdt', 'latlonSARcossin', 'todSAR',
                 'incidence', 'satellite','oswQualityCrossSpectraRe','oswQualityCrossSpectraIm']
    additional_vars_for_validation = ['oswLon','oswLat','oswLandFlag','oswIncidenceAngle']
    varstoadd += additional_vars_for_validation
    if 'time' in ds:
        newds['time'] = ds['time']
    else:
        newds['time'] = xarray.DataArray(np.array([fdatedt]),dims=['time'],coords={'time':[0]})
    for vv in varstoadd:

        if vv in ['cwave']:
            dimszi = ['time','cwavedim']
            coordi= {'time':[fdatedt],'cwavedim':np.arange(22)}
            cwave = np.hstack([S.T, s0.reshape(-1,1), nv.reshape(-1,1)]) #found L77 in preprocess.py
            cwave = preprocess.conv_cwave(cwave)
            newds[vv] = xarray.DataArray(data=cwave,dims=dimszi,coords=coordi)
        elif vv == 'S': #to ease the comparison with Justin files
            dimszi = ['time','Sdim']
            coordi = {'time' : [fdatedt],'Sdim' : np.arange(20)}
            newds[vv] = xarray.DataArray(data=S.T,dims=dimszi,coords=coordi)
        elif vv in ['dxdt']: #dx and dt and delta from coloc with alti see /home/cercache/users/jstopa/sar/empHs/cwaveV5, I can put zeros here at this stage
            #dx = preprocess.conv_dx(fs['dx'][indices])
            #dt = preprocess.conv_dt(fs['dt'][indices])
            dx = np.array([0])
            dt = np.array([1])
            dxdt = np.column_stack([dx, dt])
            dimszi = ['time','dxdtdim']
            coordi= {'time':[fdatedt],'dxdtdim':np.arange(2)}
            #print('dxdt')
            newds[vv] = xarray.DataArray(data=dxdt,dims=dimszi,coords=coordi)
        elif vv in ['latlonSARcossin']:
            latSARcossin = preprocess.conv_position(subset_ok['latSAR']) # Gets cos and sin
            lonSARcossin = preprocess.conv_position(subset_ok['lonSAR'])
            latlonSARcossin = np.hstack([latSARcossin, lonSARcossin])
            dimszi = ['time','latlondim']
            coordi= {'time':[fdatedt],'latlondim':np.arange(4)}
            newds[vv] = xarray.DataArray(data=latlonSARcossin,dims=dimszi,coords=coordi)
        elif vv in ['todSAR']:
            dimszi = ['time']
            coordi= {'time':[fdatedt]}
            newds[vv] = xarray.DataArray(data=subset_ok['todSAR'],dims=dimszi,coords=coordi)
        elif vv in ['incidence',]:
            dimszi = ['time','incdim']
            coordi= {'time':[fdatedt],'incdim':np.arange(2)}
            incidence = preprocess.conv_incidence(ds['oswIncidenceAngle'].values.squeeze())
            newds[vv] = xarray.DataArray(data=incidence,dims=dimszi,coords=coordi)
        elif vv in ['satellite']:
            dimszi = ['time']
            coordi= {'time':[fdatedt]}
            satellite_int = np.array([satellite[2] == 'a']).astype(int)
            newds[vv] = xarray.DataArray(data=satellite_int,dims=dimszi,coords=coordi)
        elif vv in ['oswQualityCrossSpectraRe','oswQualityCrossSpectraIm']:
            datatmp = ds[vv].values.squeeze()
            olddims = [x for x in ds[vv].dims if x not in ['oswAzSize','oswRaSize']]
            coordi = {}
            for didi in olddims:
                coordi[didi] = ds[vv].coords[didi].values
            coordi['time'] = [fdatedt]
            dimsadd= ['time','oswAngularBinSize','oswWavenumberBinSize']
            datatmp = datatmp.reshape((1,72,60))
            newds[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        else:
            datatmp = ds[vv].values.squeeze()
            olddims = [x for x in ds[vv].dims if x not in ['oswAzSize','oswRaSize']]
            coordi = {}
            for didi in olddims :
                coordi[didi] = ds[vv].coords[didi].values
            coordi['time'] = [fdatedt]
            dimsadd = ['time']
            newds[vv] = xarray.DataArray(data=[datatmp],dims=dimsadd,coords=coordi)

    return newds



