"""

Antoine Grouazel
27Dec2020
note premier test: je narrive pas a lire la variable S (avec les 20 cwave params) de Justin, tested avec netCDF4 et xarray... je ne vois pas comment proceder
"""
import logging
import os
import sys
import traceback
import glob
import xarray
import numpy as np
import datetime
import pdb
sys.path.append('/home1/datahome/agrouaze/git/SAR-Wave-Height/')
from load_quach_2020_keras_model import POSSIBLES_MODELS
#DIR_REF = '/home/cercache/users/jstopa/sar/empHs/forAG' #corrupted S variable
DIR_REF = '/home1/datawork/agrouaze/data/sentinel1/cwave/reference_input_output_dataset_from_Jstopa_quach2020/' #corrected S variable in

import xarray
import datetime
import netCDF4
from sarhs import preprocess



def preproc_ref_input(ds):
    """

    :param ds:
    :return:
    """
    filee = ds.encoding["source"]
    logging.info('filee %s',os.path.basename(filee))
    fdate = ds['timeSAR'].values
    try:
        fdatedt = netCDF4.num2date(fdate,ds['timeSAR'].units)
    except:
        fdatedt = fdate
    logging.info('fdatedt : %s',fdatedt)

    real_dates = []
    filesL2 = ds['fileNameL2'].values
    for tt in range(len(ds['timeSAR'])) :
        if tt % 10000 == 0 :
            print(tt,'/',len(ds['timeSAR']))
        fileL2 = filesL2[tt]
        fileL2 = fileL2.decode()
        dt = datetime.datetime.strptime(os.path.basename(fileL2).split('-')[4],'%Y%m%dt%H%M%S')
        # print(dt)
        real_dates.append(dt)

    #fdate = datetime.datetime.strptime(os.path.basename(filee).split('-')[4],'%Y%m%dt%H%M%S')
    #ds['time'] = xarray.DataArray([fdate],dims=['time'])
    #ds = ds.sortby('time',ascending=True)
    newds = xarray.Dataset()

    #format data for CWAVE 22 params computation
    #cspcRe = ds['oswQualityCrossSpectraRe'].values.squeeze().T
    #cspcIm = ds['oswQualityCrossSpectraIm'].values.squeeze().T
    #ths1 = ds['oswPhi'].values.squeeze()
    #ks1 = ds['oswK'].values.squeeze()
    #ta = ds['oswHeading'].values.squeeze()
    #incidenceangle =ds['oswIncidenceAngle'].values.squeeze()
    s0 =  ds['sigma0'].values.squeeze()
    nv = ds['normalizedVariance'].values.squeeze()
    nv = nv.reshape((len(nv),1)) #to allow concatenation with 2D S variable
    s0 = s0.reshape((len(s0),1))
    logging.info('s0: %s',s0.shape)
    logging.info('ds[S] %s %s',ds['S'].shape,ds['S'])
    #ds['S'] = ds['S'].astype('float32',casting='unsafe')
    #nc = netCDF4.Dataset(filee) #patch because S params saved by JStopa are not readable with xarray (different dtypes)
    #S = nc.variables['S'][:,0].astype('float32')
    #logging.info('S from nc: %s',S.shape)
    #lonSAR = ds['oswLon'].values.squeeze()
    #latSAR = ds['oswLat'].values.squeeze()
    #satellite = os.path.basename(filee)[0:3]
    # subset_ok,flagKcorrupted,cspcReX,cspcImX,cspcRe,ks1,ths1,kx,ky,cspcReX_not_conservativ,S = compute_hs_total_SAR_v2.format_input_CWAVE_vector_from_OCN(cspcRe,
    #                                                                         cspcIm,ths1,ta,incidenceangle,s0,nv,ks1,fdate,lonSAR,latSAR,satellite)

    varstoadd = ['cwave', 'dxdt', 'latlonSARcossin', 'todSAR', 'incidence', 'satellite','cspcRe','cspcIm','hsNN','hsNNSTD']
    #additional_vars_for_validation = ['oswLon','oswLat','oswLandFlag','oswIncidenceAngle']
    #varstoadd += additional_vars_for_validation
    newds['timeSAR'] = xarray.DataArray(fdate,dims=['time'],coords={'time':fdate})
    #newds['timeSARdt'] = xarray.DataArray(fdatedt,dims=['time'],coords={'time':fdate})
    newds['timeSARdt'] = xarray.DataArray(real_dates,dims='time',coords={'time':fdate})
    if 'S1A' in filee:
        satellite = 0
    else:
        satellite = 1
    logging.info('newds with only time: %s',newds)
    for vv in varstoadd:
        logging.info('vv : %s',vv)
        if vv in ['cwave']:
            dimszi = ['time','cwavedim']
            coordi= {'time':fdate,'cwavedim':np.arange(22)}
            #tmptmp = ds['S'].astype('float32',casting='unsafe').values[:,1]
            tmptmp = ds['S'].values
            logging.info('tmptmp : %s %s %s',tmptmp.shape,type(tmptmp),tmptmp.dtype)
            logging.info('s0 %s',s0.shape)
            logging.info('nV : %s',nv.shape)
            cwave = np.hstack([tmptmp, s0, nv]) #found L77 in preprocess.py
            logging.info('cwave : %s',cwave.shape)
            cwave = preprocess.conv_cwave(cwave)
            logging.info('cwave after normalization : %s,%s',cwave.shape,type(cwave))
            newds[vv] = xarray.DataArray(cwave,coords=coordi,dims=dimszi)
        elif vv in ['dxdt']: #dx and dt and delta from coloc with alti see /home/cercache/users/jstopa/sar/empHs/cwaveV5, I can put zeros here at this stage
            #dx = preprocess.conv_dx(fs['dx'][indices])
            #dt = preprocess.conv_dt(fs['dt'][indices])
            #dx = np.array([0])
            #dt = np.array([1])
            dx = np.zeros(len(fdate))
            dt = np.zeros(len(fdate))
            dxdt = np.column_stack([dx, dt])
            logging.info('dxdt: %s %s',dxdt.shape,dxdt)
            dimszi = ['time','dxdtdim']
            coordi= {'time':fdate,'dxdtdim':np.arange(2)}
            #print('dxdt')
            newds[vv] = xarray.DataArray(data=dxdt,dims=dimszi,coords=coordi)
        elif vv in ['latlonSARcossin']:
            latSARcossin = preprocess.conv_position(ds['latSAR']) # Gets cos and sin
            lonSARcossin = preprocess.conv_position(ds['lonSAR'])
            latlonSARcossin = np.hstack([latSARcossin, lonSARcossin])
            dimszi = ['time','latlondim']
            coordi= {'time':fdate,'latlondim':np.arange(4)}
            newds[vv] = xarray.DataArray(data=latlonSARcossin,dims=dimszi,coords=coordi)
        elif vv in ['todSAR']:
            dimszi = ['time']
            coordi= {'time':fdate}
            todSAR = preprocess.conv_time(fdate)
            logging.info('todSAR : %s',todSAR)
            newds[vv] = xarray.DataArray(data=todSAR,dims=dimszi,coords=coordi)
        elif vv in ['incidence',]:
            dimszi = ['time','incdim']
            coordi= {'time':fdate,'incdim':np.arange(2)}
            incidence = preprocess.conv_incidence(ds['incidenceAngle'].values.squeeze())
            newds[vv] = xarray.DataArray(data=incidence,dims=dimszi,coords=coordi)
        elif vv in ['satellite']:
            dimszi = ['time']
            coordi= {'time':fdate}
            #satellite_int = np.array([satellite[2] == 'a']).astype(int)
            #satellite_int = np.repeat(satellite_int,len(fdate))
            satellite_int = np.ones((ds['timeSAR'].shape[0], ), dtype=float) * satellite
            logging.info('satellite_int = %s',satellite_int.shape)
            newds[vv] = xarray.DataArray(data=satellite_int,dims=dimszi,coords=coordi)
        elif vv in ['cspcRe','cspcIm']:
            datatmp = ds[vv].values.squeeze()
            logging.info('vv: %s shape : %s',vv,datatmp.shape)
            olddims = [x for x in ds[vv].dims if x not in ['oswAzSize','oswRaSize']]
            coordi = {}
            for didi in olddims:
                coordi[didi] = ds[vv].coords[didi].values
            coordi['time'] = fdate
            dimsadd= ['time','directions','wavenumbers']
            #datatmp = datatmp.reshape((1,72,60))
            newds[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)
        else:
            datatmp = ds[vv].values.squeeze()
            olddims = [x for x in ds[vv].dims if x not in ['oswAzSize','oswRaSize']]
            coordi = {}
            for didi in olddims :
                coordi[didi] = ds[vv].coords[didi].values
            coordi['time'] = fdate
            dimsadd = ['time']
            newds[vv] = xarray.DataArray(data=datatmp,dims=dimsadd,coords=coordi)

    return newds

def read_input_files(single_input_ref_file):
    """
    dataset provided by J stopa 16 dec 2020 (input and output are in the same files
    :return:
    """

    logging.info('arbitrary chosen input ref file : %s',single_input_ref_file)
    dsref0 = xarray.open_mfdataset(single_input_ref_file,preprocess=preproc_ref_input,decode_times=False)
    #pdb.set_trace()
    dsref = dsref0.where(dsref0['timeSARdt']<np.datetime64('2019-06-02'),drop=True) #for test I only take the first day
    #pdb.set_trace()
    #dsref = dsref0.where(dsref0['timeSARdt']<datetime.datetime(2019,1,2),drop=True)
    logging.info('timeSAR : %s',dsref['timeSAR'].size)
    cspcRe = dsref['cspcRe'].values.squeeze()
    cspcIm = dsref['cspcIm'].values.squeeze()
    re = preprocess.conv_real(cspcRe)
    im = preprocess.conv_imaginary(cspcIm)
    spectrum = np.stack((re,im),axis=3)
    logging.info('spectrum shape : %s',spectrum.shape)
    return spectrum,dsref

def define_features(ds):
    """
    :param ds: xarrayDataArray of reference WV data
    :return:
    """
    features = None
    for jj in ['cwave','dxdt','latlonSARcossin','todSAR','incidence','satellite'] :
        addon = ds[jj].values
        if len(addon.shape) == 1 :
            addon = addon.reshape((addon.size,1))
        if features is None :
            features = addon
        else :
            features = np.hstack([features,addon])
    #  = np.dstack([ds['cwave'].values,ds['dxdt'].values,ds['latlonSARcossin'].values,ds['todSAR'].values,ds['incidence'].values,ds['satellite'].values])
    logging.info('features ready')
    return features

def define_input_test_dataset(features,spectrum):
    """

    :param features: 2D np matrix
    :param spectrum: 4D np matrix
    :return:
    """
    outputs = np.zeros(features.shape[0])
    inputs = [spectrum,features]
    test = (inputs,outputs)
    logging.info('test dataset ready')
    return test

def predict ( model,dataset ) :
    """

    :param model:
    :param dataset:
    :return:
    """
    ys,yhats = [],[]
    for batch in dataset :
        inputs,y = batch
        print('inputs',type(inputs))
        print('y',y)
        yhat = model.predict_on_batch(inputs)
        if y is not None :
            y = y.reshape(-1,2)
        else :
            y = np.zeros((yhat.shape[0],1))
        ys.append(y)
        yhats.append(yhat)
    yhat = np.vstack(yhats)
    y = np.vstack(ys)
    return y,yhat

# my version without batch
def predict_agrouaze ( model,dataset ) :
    """

    :param model:
    :param dataset:
    :return:
    """
    ys,yhats = [],[]
    inputs,yhats_vide = dataset
    #yhat = model.predict_on_batch(inputs)
    yhat = model.predict(inputs)
    logging.debug('yhat %s %s',yhat.shape,yhat)
    # if y is not None :
    #     y = y.reshape(-1,2)
    #     pass
    # else :
    #     y = np.zeros((yhat.shape[0],1))
    # ys.append(y)
    yhats.append(yhat)
    yhat = np.vstack(yhats)
    #y = np.vstack(ys)
    return yhat

def do_my_prediction(model,test):
    """

    :param model:
    :param test:
    :return:
    """
    from tqdm import tqdm
    logging.info('start Hs prediction')
    # _, yhat = predict(model,test)
    yhat = predict_agrouaze(model,test)
    logging.info('prediction finished')
    return yhat


def main_level_0(MODEL,ref_ds,spectrum):
    """

    :param ref_ds: xarray data array read by prepare_ocn_wv_data()
    :return:
    """
    output_prediction = {}
    features = define_features(ref_ds)
    test = define_input_test_dataset(features,spectrum)
    yhat = do_my_prediction(MODEL,test)
    output_prediction['HsQuach'] = xarray.DataArray(data=yhat[:,0],dims=['time'],coords={'time':ref_ds['timeSAR'].values})
    output_prediction['HsQuach_uncertainty'] = xarray.DataArray(data=yhat[:,1],dims=['time'],coords={'time':ref_ds['timeSAR'].values})
    return output_prediction,features

def predict_and_save(ref_ds,outputdir,hs_ref,hs_ref_std,input_ref_file,modelname,featuresArray):
    """

    :param ref_ds: xarray dataset
    :param outputdir: str
    :param hs_ref: ndarray 1D
    :param hs_ref_std: ndarray 1D
    :param input_ref_file: str
    :param modelname: str
    :param featuresArray (nd array) nbobs X 32 params
    :return:
    """

    ref_ds['hs_ref_ifr'] = xarray.DataArray(hs_ref,dims=['time'])
    ref_ds['hs_ref_std_ifr'] = xarray.DataArray(hs_ref_std,dims=['time'])
    ref_ds['features'] = xarray.DataArray(featuresArray,dims=['time','featdim'],coords={'time':ref_ds['time'].values,'featdim':np.arange(32)})
    logging.debug('ref_ds modified : %s',ref_ds)
    #pdb.set_trace()
    outputpath = os.path.join(outputdir,'Quach2020_ifr_predict_using_ref_CWAVE_%s_%s'%(modelname,os.path.basename(input_ref_file)))
    ref_ds.to_netcdf(outputpath)
    logging.info('output : %s',outputpath)
    logging.info('finished')

def choose_model(args):
    MODEL = None
    MODEL = POSSIBLES_MODELS[args.modelversion]()
    # if args.modelversion == 'justin_std' :
    #     MODEL = load_quach2020_model()
    # elif args.modelversion == 'heteroskadastic' :
    #     MODEL = load_quach2020_model_v2()
    # elif args.modelversion == 'justin_basic' :#ne marche pas car pas de prediction sur uncertainty
    #     MODEL = load_quach2020_model_basic()
    # elif args.modelversion == 'only_std' :
    #     MODEL = load_quach2020_model_45_std_tuned()
    # else :
    #     raise Exception('model %s unknown' % args.modelversion)
    return MODEL

if __name__ =='__main__':
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='basic test to solve the loading')
    parser.add_argument('--verbose',action='store_true',default=False)

    parser.add_argument('--modelversion',action='store',choices=POSSIBLES_MODELS.keys(),required=True,help='possible models: %s'%POSSIBLES_MODELS.keys())

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')

    print('ok')
    logging.info('start prediction with model : %s',args.modelversion)
    logging.info('done')
    one_file = os.path.join(DIR_REF,'S1A_201901S.nc')
    spectrum,ref_ds = read_input_files(one_file)
    logging.info('preprocessing done')
    try :
        print('first try to  load_quach2020_model')
        MODEL = choose_model(args)
    except :
        logging.error('trace : %s',traceback.format_exc())
        print('second try to  load_quach2020_model (except)')#the second try his here because the loading with justin_std have a problem of layer naming solved by two consecutive call
        MODEL = choose_model(args)
    output_prediction,featuresArray =  main_level_0(MODEL,ref_ds,spectrum)
    outputdir = '/home1/datawork/agrouaze/data/sentinel1/cwave/validation_quach2020/'
    logging.info('outputdir : %s',outputdir)
    predict_and_save(ref_ds,outputdir,hs_ref=output_prediction['HsQuach'],
                     hs_ref_std=output_prediction['HsQuach_uncertainty'],input_ref_file=one_file,
                     modelname=args.modelversion,featuresArray=featuresArray)
    logging.info('done')
