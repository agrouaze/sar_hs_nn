"""
July 2022
Grouazel
"""
import os
import sys
import logging
import xarray
from dateutil import rrule
import numpy as np
import glob
import datetime
import time
import netCDF4
from normalization_inputs_exp1v4_nc_outputs import kref
from sarhspredictor.lib.sarhs import preprocess

def prep(ds):
    #ds = ds.drop('py_S') #bug on the run for 2015 16h the 2 June 2021
    filee = ds.encoding['source']
    if 'dk' in  ds:
        ds = ds.drop('dk')
    if 'k' in ds:
        ds = ds.drop('k')
    ds = ds.rename( {'time_sar':'time'})
    return ds

def normalize_training_ds(sta,sto,in_dd,out_dd,redo=False,dev=False):
    """

    :param sta: datetime
    :param sto: datetime
    :param in_dd: str where the colocs SAR and Alti are stored
    :param out_dd: str where the same dataset but normalized will be stored
    :return:
    """
    if os.path.exists(out_dd) is False:
        os.makedirs(out_dd,0o0775)
        logging.info('makedir %s',out_dd)
    for sat in ['S1A','S1B']:
        if sat=='S1A':
            satellite = 1 # 1=S1A, 0=S1B
        else:
            satellite = 0
        for dd in rrule.rrule(rrule.DAILY,dtstart=sta,until=sto):
            pat = os.path.join(in_dd,dd.strftime('%Y'),'*',
                               'training_' + '*%s*%s.nc' % (dd.strftime('%Y%m%d'),sat))
            #file_dest = os.path.join(out_dd,dd.strftime('%Y'),sat+'_training_exp1v4_Hs_NN_regression_%s.nc'%dd.strftime('%Y%m%d'))
            file_dest = os.path.join(out_dd, dd.strftime('%Y'),
                                     sat + '_training_exp2_Hs_NN_regression_%s.nc' % dd.strftime('%Y%m%d'))
            if os.path.exists(file_dest) and redo is True:
                os.remove(file_dest)
                logging.info('remove existing output file : %s',file_dest)
            if os.path.exists(file_dest) and redo is False:
                logging.info('%s already exists -> skip and continue the daily loop',file_dest)
                continue
            lst_files_measu_to_read = glob.glob(pat)#[0:3000] # for dev !!!!!!!!!!!
            if dev:
                lst_files_measu_to_read = lst_files_measu_to_read[0:4]
            logging.info('nb files for daily %s : %s',dd,len(lst_files_measu_to_read))
            if len(lst_files_measu_to_read)>0:
                #print('file_dest',file_dest,'daily',dd)
                if os.path.exists(os.path.dirname(file_dest)) is False:
                    os.makedirs(os.path.dirname(file_dest))
                    #print('outputdir mkdir')
                # These variables are expected in the source file.
                keys = ['time', 'lonSAR',  'latSAR', 'angle_of_incidence', 'crossSpectraRePol', 'crossSpectraImPol'
                        ,'s0_SLC','nv_SLC'] # Needed for predictions.
                #keys_ocn = ['S','cspcRe','cspcIm']
                keys = keys # added to have both slc and ocn in the same dataset (July 21)
                t0 = time.time()
                # try:
                #     h5py.File(file_dest, 'r').close() #try to close the file if it is opened before
                # except:
                #     print('traceback',traceback.format_exc())
                #     pass
                src = xarray.open_mfdataset(lst_files_measu_to_read,concat_dim='time',combine='nested', #,combine='by_coords'
                                            preprocess=prep,cache=True,decode_times=True) #,preprocess=None


                if len(src['time'])>0:
                    src = src.assign_coords({'k':kref})
                    logging.info('src: %s',src)
                    #with Dataset(file_src) as fs, h5py.File(file_dest, 'w') as fd:
                    # Check input file.
                    #src = fs.variables
                    destds = xarray.Dataset()
                    #with h5py.File(file_dest, 'w') as fd:
                    for k in keys:
                        if k not in src:
                            raise IOError(f'Variable {k} not found in input file.')
                    num_examples = src[keys[0]].shape[0]
                    print(f'Found {num_examples} events.')

                    # Get 22 CWAVE features. Concatenate 20 parameters with sigma0 and normVar.
                    #src['S'].set_auto_scale(False) # Some of the NetCDF4 files had some weird scaling.
                    S = src['CWAVE_20_SLC'].values#.reshape(-1,20) #* float(src['py_S'].scale_factor))
                    logging.info('s0_SLC %s',src['s0_SLC'].values.reshape(-1,1).shape)
                    logging.info('nv_SLC %s',src['nv_SLC'].values.reshape(-1,1).shape)
                    logging.info('S_slc %s',S.shape)
                    cwave_slc = np.hstack([S, src['s0_SLC'].values.reshape(-1,1), src['nv_SLC'].values.reshape(-1,1)])
                    #cwave = src['cwave'][:]
                    cwave_slc = preprocess.conv_cwave(cwave_slc) # Remove extrema, then standardize with hardcoded mean, vars.
                    #fd.create_dataset('cwave', data=cwave)

                    #add cwave from OCN (to comapre)
                    #Socn = np.array(src['S'].values)  # * float(src['py_S'].scale_factor))
                    #cwave_ocn = np.hstack(
                    #    [Socn,src['sigma0'].values.reshape(-1,1),src['normalizedVariance'].values.reshape(-1,1)])
                    # # cwave = src['cwave'][:]
                    #cwave_ocn = preprocess.conv_cwave(cwave_ocn)  # Remove extrema, then standardize with hardcoded mean, vars.
                    # fd.create_dataset('cwave_ocn',data=cwave_ocn)


                    # Observation meta data.
                    latSAR, lonSAR = src['latSAR'].values, src['lonSAR'].values
                    latSARcossin = preprocess.conv_position(latSAR) # Computes cos and sin used by NN.
                    lonSARcossin = preprocess.conv_position(lonSAR)
                    #latlonSARcossin = src['latlonSARcossin'].values
                    #fd.create_dataset('latlonSAR', data=np.column_stack([latSAR, lonSAR]))
                    destds['latlonSAR'] = xarray.DataArray(np.column_stack([latSAR, lonSAR]),
                                                        coords={'nsample':src['time'].values,'lonlat':np.arange(2)},
                                                           dims=['nsample','lonlat'],
                                                           attrs={'description':'lat,lon from SAR WV','units':'degrees North'}
                                                           )
                    #fd.create_dataset('latlonSARcossin', data=np.hstack([latSARcossin, lonSARcossin]))
                    destds['latlonSARcossin_normed'] = xarray.DataArray(np.hstack([latSARcossin, lonSARcossin]),
                                                        coords={'nsample':src['time'].values,'latSARcossinlonSARcossin':np.arange(4)},
                                                           dims=['nsample','latSARcossinlonSARcossin'],
                                                                        attrs={'description': 'normed cos(lat),sin(lat),cos(lon),sin(lon) from SAR WV',
                                                                               'units': ''}
                                                        )
                    destds['lonsar_SLC'] = xarray.DataArray(src['lonsar_SLC'],coords={'nsample':src['time'].values},dims=['nsample'],
                                                         attrs={
                                                             'description': 'longitude of the center of the WV image annotated in SLC product',
                                                             'min':-180,
                                                             'max':180
                                                             }
                                                         )
                    destds['latsar_SLC'] = xarray.DataArray(src['latsar_SLC'],
                                                            coords={'nsample': src['time'].values}, dims=['nsample'],
                                                            attrs={
                                                                'description': 'latitude of the center of the WV image annotated in SLC product',
                                                                'min':-90,
                                                                'max':90

                                                            }
                                                            )
                    #fd.create_dataset('latlonSARcossin', data=latlonSARcossin)

                    time_sar = src['time'].values

                    unit = 'hours since 2010-01-01T00:00:00Z UTC' #asked by P.Sadowsky routine
                    logging.info("time units: %s",unit)


                    # idem for timeALT
                    timeALT = src['time_ALTI'].values  # added by agrouaze
                    time_num = []
                    for tti in timeALT :
                        ts = (tti - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
                        ts2 = datetime.datetime.utcfromtimestamp(ts)
                        tnum = netCDF4.date2num(ts2,unit)
                        time_num.append(tnum)
                    timeALT = np.array(time_num)

                    logging.info('time_sar first value = %s %s',time_sar[0],type(time_sar[0]))
                    doySAR = preprocess.conv_time_doy(time_sar)
                    time_num = []
                    for tti in time_sar:
                        ts = (tti - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                        ts2 = datetime.datetime.utcfromtimestamp(ts)
                        tnum = netCDF4.date2num(ts2, unit)
                        time_num.append(tnum)
                    timeSARnum = np.array(time_num)
                    toDSAR = preprocess.conv_time(timeSARnum)
                    destds['doySAR_normed'] = xarray.DataArray(doySAR,coords={'nsample':src['time'].values},dims=['nsample'],
                                                         attrs={
                                                             'description': 'day of year (between 1 and 365) divided by 365',
                                                             }
                                                         )
                    destds['todSAR_normed'] = xarray.DataArray(toDSAR,coords={'nsample':src['time'].values},dims=['nsample'],
                                                         attrs={
                                                             'description': 'hour of the day SAR',
                                                             }
                                                         )
                    #todSAR = src['todSAR'].values

                    destds['timeSAR'] = xarray.DataArray(time_sar,coords={'nsample':src['time'].values},dims=['nsample'],
                                                         attrs={
                                                             'description': 'start date of the WV acquisition',
                                                             }
                                                         )
                    #fd.create_dataset('doySAR', data=doySAR, shape=(doySAR.shape[0], 1))

                    # incidence = preprocess.conv_incidence_iw(src['angle_of_incidence'].values) # Separates into 2 var.
                    # destds['incidence_OCN_normed'] = xarray.DataArray(incidence, coords={'nsample': src['time'].values}, dims=['nsample'],
                    #                                        attrs={
                    #                                            'description': 'normed incidence angle from SAR WV oswIncidenceAngle',
                    #                                            'units': ''}
                    #                                        )
                    incidence_slc = preprocess.conv_incidence_iw(src['incidenceangle_SLC'].values)  # Separates into 2 var.
                    destds['incidence_SLC_normed'] = xarray.DataArray(incidence_slc, coords={'nsample': src['time'].values},
                                                           dims=['nsample'],
                                                           attrs={
                                                               'description': 'normed incidence angle from SAR WV SLC',
                                                               'units': ''}
                                                           )
                    #fd.create_dataset('incidence', data=incidence)
                    # destds['trackAngle'] = xarray.DataArray(src['trackAngle'], coords={'nsample': src['time'].values},
                    #                                       dims=['nsample'],
                    #                                       attrs={
                    #                                           'description': 'SAR WV oswHeading from OCN ',
                    #                                           'units': 'degrees clockwise wrt North'}
                    #                                       )
                    destds['trackAngle_SLC'] = xarray.DataArray(src['ta_SLC'].values,
                                                            coords={'nsample': src['time'].values},
                                                            dims=['nsample'],
                                                            attrs={'description': 'track angle annotation from WV SLC',
                                                                   'units': 'degrees clockwise wrt North'})


                    # Spectral data.
                    logging.debug('x spec Re: %s',src['crossSpectraRePol'].values.shape)
                    tmpre = src['crossSpectraRePol'].values#.squeeze()
                    tmpre = np.swapaxes(tmpre,1,2)
                    tmpim = src['crossSpectraImPol'].values#.squeeze()
                    tmpim = np.swapaxes(tmpim,1,2)
                    logging.debug('tmpre : %s',tmpre.shape)
                    re = preprocess.conv_real(tmpre,exp_id=1)
                    im = preprocess.conv_imaginary(tmpim,exp_id=1)
                    x = np.stack((re, im), axis=3)
                    logging.debug('spectrum : %s',x.shape)
                    #fd.create_dataset('spectrum', data=x) # spectrum doesnt seems to be used at the next preprocessing step ( aggregate_monthly_training_files.aggregate() )
                    destds['spectrum_slc_normed'] = xarray.DataArray(x, coords={'nsample': src['time'].values,
                                                                            'k':kref,'phi':np.arange(0,360,5),
                                                                            're_im':np.arange(2)},
                                                           dims=['nsample','phi','k','re_im'],
                                                                 attrs={
                                                                     'description': 'normed real and imaginary part of the image cross spectra from SLC SAR WV',
                                                                     'units': ''}
                                                                 )
                    # add spectral data OCN
                    # logging.debug('x spec Re: %s',src['cspcRe'].values.shape)
                    # tmpre = src['cspcRe'].values.squeeze()
                    # #tmpre = np.swapaxes(tmpre,1,2)
                    # tmpim = src['cspcIm'].values.squeeze()
                    # #tmpim = np.swapaxes(tmpim,1,2)
                    # logging.debug('tmpre : %s',tmpre.shape)
                    # re_ocn = preprocess.conv_real(tmpre,exp_id=1)
                    # im_ocn = preprocess.conv_imaginary(tmpim,exp_id=1)
                    # x_ocn = np.stack((re_ocn,im_ocn),axis=3)
                    # # logging.debug('spectrum : %s',x_ocn.shape)
                    #
                    # # fd.create_dataset('spectrum_ocn',
                    # #                   data=x_ocn)  # spectrum doesnt seems to be used at the next preprocessing step ( aggregate_monthly_training_files.aggregate() )
                    # destds['spectrum_ocn_normed'] = xarray.DataArray(x_ocn, coords={'nsample': src['time'].values,
                    #                                                         'k':kref,'phi':np.arange(0,360,5),
                    #                                                         're_im':np.arange(2)},
                    #                                        dims=['nsample','phi','k','re_im'],
                    #                                              attrs={
                    #                                                  'description': 'normed real and imaginary part of the image cross spectra from OCN SAR WV (computed with python libraries)',
                    #                                                  'grid':'polar k/phi',
                    #                                                  'units': ''}
                    #                                              )
                    # Altimeter features.
                    hsALT = src['hs_alti_mean'].values
                    #fd.create_dataset('hsALT', data=hsALT, shape=(hsALT.shape[0], 1))
                    destds['hsALT'] = xarray.DataArray(hsALT, coords={'nsample': src['time'].values},
                                                           dims=['nsample'],
                                                       attrs={'description':'Hs from colocated altimeters','units':'m'})
                    # ajout Oct 2021: wind speed altimetric
                    # wsALT = src['wsALT'].values
                    # #fd.create_dataset('wsALT',data=wsALT,shape=(wsALT.shape[0],1))
                    # destds['wsALT'] = xarray.DataArray(wsALT, coords={'nsample': src['time'].values},
                    #                                    dims=['nsample'],
                    #                                    attrs={'description': 'wind speed from colocated altimeters',
                    #                                           'units': 'm/s'})
                    #dx = preprocess.conv_dx(src['dx'].values)
                    #dt = preprocess.conv_dt(src['dt'].values)
                    #fd.create_dataset('dxdt', data=np.column_stack([dx, dt]))


                    #fd.create_dataset('timeALT',data=timeALT, shape=(doySAR.shape[0], 1))
                    destds['timeALT'] = xarray.DataArray(timeALT, coords={'nsample': src['time'].values},
                                                       dims=['nsample'],
                                                       attrs={'description': 'date of altimeters products'})

                    lonALT = src['lon_ALT'].values #added by agrouaze
                    #fd.create_dataset('lonALT', data=lonALT)
                    destds['lonALT'] = xarray.DataArray(lonALT, coords={'nsample': src['time'].values},
                                                         dims=['nsample'],
                                                         attrs={'description': 'longitude of altimeters'})

                    latALT = src['lat_ALT'].values #added by agrouaze
                    #fd.create_dataset('latALT', data=latALT)
                    destds['latALT'] = xarray.DataArray(latALT, coords={'nsample': src['time'].values},
                                                        dims=['nsample'],
                                                        attrs={'description': 'latitude of altimeters'})

                    #fd.create_dataset('hsSM', data=src['hsSM'].values) #added by agrouaze
                    #fd.create_dataset('nk', data=src['nk'].values) #added by agrouaze
                    # destds['nk'] = xarray.DataArray(src['nk'].values, coords={'nsample': src['time'].values},
                    #                                     dims=['nsample'],
                    #                                     attrs={'description': 'number of points in the wave number vector decribing the cross spectra'})
                    #fd.create_dataset('dx', data=src['dx'].values) #added by agrouaze
                    #fd.create_dataset('dt', data=src['dt'].values) #added by agrouaze
                    destds['dx'] = xarray.DataArray(src['delta_d_closest'].values, coords={'nsample': src['time'].values},
                                                    dims=['nsample'],
                                                    attrs={
                                                        'description': 'delta spatial between WV image center and alti closest center footprint',
                                                        'unit':'km'})
                    destds['dt'] = xarray.DataArray(src['delta_t_closest'].values, coords={'nsample': src['time'].values},
                                                    dims=['nsample'],
                                                    attrs={
                                                        'description': 'delta time betwenn start of WV acquisition and timestamp of the closest altimeter footprint',
                                                        'unit':'s'})
                    #fd.create_dataset('sigma0', data=src['sigma0'].values) #added by agrouaze
                    # destds['sigma0'] = xarray.DataArray(src['sigma0'].values, coords={'nsample': src['time'].values},
                    #                                 dims=['nsample'],
                    #                                 attrs={'description': 'denoised sigma0 from WV product OCN','units':'dB'})
                    destds['sigma0_SLC'] = xarray.DataArray(src['s0_SLC'].values, coords={'nsample': src['time'].values},
                                                        dims=['nsample'],
                                                        attrs={'description': 'denoised sigma0 from WV product SLC',
                                                               'units': 'dB'})
                    #fd.create_dataset('normalizedVariance', data=src['normalizedVariance'].values) #added by agrouaze
                    # destds['normalizedVariance'] = xarray.DataArray(src['normalizedVariance'].values, coords={'nsample': src['time'].values},
                    #                                     dims=['nsample'],
                    #                                     attrs={'description': 'normalized Variance from digital number stored in WV product OCN',
                    #                                            'units': ''})
                    destds['normalizedVariance_SLC'] = xarray.DataArray(src['nv_SLC'].values,
                                                            coords={'nsample': src['time'].values},
                                                            dims=['nsample'],
                                                            attrs={'description': 'normalized Variance from digital number stored in WV product SLC',
                                                                   'units': ''})
                    #fd.create_dataset('incidenceAngle', data=src['incidenceAngle'].values) #added by agrouaze
                    destds['incidenceAngle'] = xarray.DataArray(src['angle_of_incidence'].values,
                                                                    coords={'nsample': src['time'].values},
                                                                    dims=['nsample'],
                                                                    attrs={
                                                                        'description': 'incidence angle without normalization',
                                                                        'units': 'degree'})
                    #fd.create_dataset('lonSAR', data=src['lonSAR'].values) #added by agrouaze
                    destds['lonSAR'] = xarray.DataArray(src['lonSAR'].values,
                                                                    coords={'nsample': src['time'].values},
                                                                    dims=['nsample'],
                                                                    attrs={
                                                                        'description': 'longitude at the center of WV image',
                                                                        'units': 'degree'})
                    #fd.create_dataset('latSAR', data=src['latSAR'].values) #added by agrouaze
                    destds['latSAR'] = xarray.DataArray(src['latSAR'].values,
                                                        coords={'nsample': src['time'].values},
                                                        dims=['nsample'],
                                                        attrs={
                                                            'description': 'latitude at the center of WV image',
                                                            'units': 'degree'})
                    #fd.create_dataset('cspcRe_slc', data=src['crossSpectraRePol'].values) #added by agrouaze
                    destds['cspcRe_slc'] = xarray.DataArray(src['crossSpectraRePol'].values,
                                                        coords={'nsample': src['time'].values,'k':kref,'phi':np.arange(0,360,5)},
                                                        dims=['nsample','k','phi'],
                                                        attrs={
                                                            'description': 'real part of cross spectra computed from SLC product using xsar',
                                                            'units': ''})
                    #fd.create_dataset('cspcIm_slc', data=src['crossSpectraImPol'].values) #added by agrouaze
                    destds['cspcIm_slc'] = xarray.DataArray(src['crossSpectraImPol'].values,
                                                        coords={'nsample': src['time'].values,'k':kref,'phi':np.arange(0,360,5)},
                                                        dims=['nsample','k','phi'],
                                                        attrs={
                                                            'description': 'imaginary part of cross spectra computed from SLC product using xsar',
                                                            'units': ''})

                    destds['CWAVE_SLC_normed'] = xarray.DataArray(cwave_slc,
                                                        coords={'nsample': src['time'].values,'cwave_coord':np.arange(22)},
                                                        dims=['nsample','cwave_coord'],
                                                        attrs={
                                                            'description': 'normed CWAVE parameters computed from SLC spectrum (xsar+xrft) using cartesian cross spectrum',
                                                            'units': ''})
                    glob_attrs = {'processing_method':normalize_training_ds.__name__,
                                  'processing_script':os.path.basename(__file__),
                                  'processing_env':sys.executable,
                                  'processing_date':datetime.datetime.today().strftime('%Y%m%d %H:%M')
                                  ,'input_dir':in_dd,
                                  'outputdir_dir':out_dd
                                  }
                    for uu in src.attrs:
                        glob_attrs['rebuild_step_%s'%uu] = src.attrs[uu]
                    destds.attrs = glob_attrs
                    destds.to_netcdf(file_dest,
                                     encoding={'nsample':{'units':'seconds since 2014-01-01 00:00:00'}})
                    #fd.create_dataset('cspcRe_ocn',data=src['cspcRe'].values)  # added by agrouaze
                    #fd.create_dataset('cspcIm_ocn',data=src['cspcIm'].values)  # added by agrouaze
                    #fd.create_dataset('py_S', data=S) #added by agrouaze
                    #fd.create_dataset('py_S_ocn',data=Socn)  # added by agrouaze
                    #fd.close()
                    #check open the new file
                    #test_post_write = h5py.File(file_dest,'r')
                    #test_post_write.close()
                    print('elapsed time to build %s : %1.3f seconds'%(file_dest,time.time()-t0))
                    logging.info('file written %s',os.path.exists(file_dest))

if __name__ == '__main__':
    root = logging.getLogger ()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler (handler)
    import argparse
    import resource
    time.sleep(np.random.rand(1,1)[0][0]) #to avoid issue with mkdir
    parser = argparse.ArgumentParser (description='norm inputs Hs NN training')
    parser.add_argument ('--verbose',action='store_true',default=False)
    parser.add_argument ('--startdate',action='store',help='YYYYMMDD',required=True)
    parser.add_argument ('--stopdate',action='store',help='YYYYMMDD.',required=True)
    parser.add_argument('--redo',action='store_true',default=False,required=False,help='redo existing output files')
    parser.add_argument('--dev',action='store_true',default=False,required=False,help='reduce input files to 3 files')
    args = parser.parse_args ()

    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'

    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time ()
    #in_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v2/'
    #in_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3/'
    #in_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v4/'
    #in_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v6/'
    in_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v7/' # 24 may 2022
    in_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v8/'  # see readme.txt
    in_dd = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp2D4/v2/'
    #in_dd = '/home1/scratch/agrouaze/test_D1v4' # finalement je peux repartir depuis le dataset d input D1v3, il ny a que la normalization et les variables dispo qui change...
    # out_dd = '/home1/scratch/agrouaze/training_quach_redo_model/exp1/daily_normalized'
    # out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3_norm'
    # out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3_norm_v2'
    # out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3_norm_v3'
    # out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3_norm_v4'
    # out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3_norm_v5' # 99 percentil
    # out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3_norm_v6' # 99% + correction of droped colocs
    # out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v3_norm_v7' # 99% + correction of droped + fix multi successive norm
    # out_dd = '/home1/scratch/agrouaze/test_D1v4_norm'
    # 11 oct D1v4 from raw colocs v3
    #out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v1_norm'
    out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v4_norm'
    out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v5_norm' # correction on the nsample units
    out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v6_norm'  # add todSAR,doySAR and nsample to test influence of time variables
    out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v7_norm'  # same as v6 but revised normalization
    out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v8_norm' # imaginary part keep negative and positive values (no more abs() applied)
    out_dd = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v9_norm' # all variables CWAVE , HLF, spec OCN and SLC, needed by T Grange for validating the change of spectrum only (June 2022)
    out_dd = '/home1/scratch/agrouaze/normalization_test_exp2/'
    out_dd = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp2D4/v2_normed/'
    #out_dd = '/home1/scratch/agrouaze/'
    logging.info('input dir %s',in_dd)
    logging.info('output dir : %s',out_dd)
    normalize_training_ds(sta=datetime.datetime.strptime(args.startdate,'%Y%m%d'),
                          sto=datetime.datetime.strptime(args.stopdate,'%Y%m%d'),in_dd=in_dd,out_dd=out_dd,redo=args.redo,
                          dev=args.dev)
    logging.info('done in %1.3f min',(time.time()-t0)/60.)
    logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)