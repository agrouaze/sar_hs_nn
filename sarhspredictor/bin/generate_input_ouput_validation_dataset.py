"""
Feb 2021
A Grouazel
script to get a dataset containing hs NN, S params, spectra Re Im and hs alti
    - input params are coming from ocn and CWAVE v6 datasets (only nrcs,nv, polar Xspec and timeSAR are coming from cwaveV6)
    - cross spectra are coming from cwave V6 (they are in polar grid 72x60 )
    - CWAVE are computed on the fly
    - Hs NN are predicted on the fly using previous CWAVE and cross spectra via model heteroskedastic provided by Peter Sadowsky the 4 Feb 2021
history: inspired by rebuild_training_dataset.py with main difference is that I need predictions Hs and Hs _std
10 feb 2021: modification of the script to treat the data day by day for throughput purpose (monthly = too long)
env: cwave
"""
import logging
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' #patch for datarmor
import glob
import collections
import netCDF4
import copy
import numpy as np
import sys
import resource
import time
import scipy
import pdb
import xarray
import datetime
from sarhspredictor.lib.apply_oswK_patch import patch_oswK
from sarhspredictor.lib.compute_CWAVE_params import format_input_CWAVE_vector_from_OCN
from sarhspredictor.lib.load_quach_2020_keras_model import load_quach2020_model_v2
from sarhspredictor.lib.predict_with_quach2020_on_OCN_using_keras import do_my_prediction,define_features,define_input_test_dataset
from sarhspredictor.lib.predict_hs_from_training_dataset import prepare_training_dataset_core,compute_prediction_from_training_inputs
sciV = scipy.__version__
numV = np.__version__
pyv = sys.version_info
#OUTPUTDIR = '/home1/datawork/agrouaze/data/sentinel1/cwave/validation_quach2020/heteroskedastic2017_version_4feb2021/'
OUTPUTDIR_TMP = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/quach2020/validation/input_output/tmp'
OUTPUTDIR_FINAL = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/quach2020/validation/input_output/final'
INPUT_JUSTIN = '/home/cercache/users/jstopa/sar/empHs/cwaveV6' #v6 contains year 2019 out of training dataset
INPUT_JUSTIN = '/home1/datawork/agrouaze/data/sentinel1/cwave/reference_input_output_dataset_from_Jstopa_quach2020' #this dataset is the same but with S fixed... there should be not difference at all because S is compute here on the fly

def append_python_vars_ds2019 ( satellite,datedaydt,ffout,dev=False ) :
    """
    10 fev 2021: I spit the dataset by day to distribute otherwise monthly chuncks take too much time to process
    :param satellite: S1A or S1B (str)
    :param datedaydt: datetime
    :param dev:
    :return:
    """

    ff = os.path.join(INPUT_JUSTIN,satellite+'_'+datedaydt.strftime('%Y%m')+'S.nc')
    logging.info('treat %s',ff)
    nci = netCDF4.Dataset(ff,'r')
    # get the right L2 paths
    tmp_values = []
    dates_sar = []
    filenames_L2 = nci.variables['fileNameL2'][:]
    for iiu in range(filenames_L2.shape[0]) :
        apath = ('').join([ddc.decode() for ddc in filenames_L2[iiu,:]])
        tmp_values.append(apath.replace('cercache','datawork-cersat-public'))
        dates_sar.append(datetime.datetime.strptime(os.path.basename(apath).split('-')[4],'%Y%m%dt%H%M%S'))
    dates_sar = np.array(dates_sar)
    datedaydt_plus_24h = datedaydt + datetime.timedelta(hours=24)
    dates_indice_ok = np.where((dates_sar>=datedaydt) & (dates_sar<datedaydt_plus_24h))[0]

    L2_paths_datarmor = np.array(tmp_values)  # [0:N_time]

    if dev is True :

        N_time = 10 #10 previously
        logging.warning('for dev N_time = %s',N_time)
        sli = slice(0,N_time)
        dates_indice_ok = dates_indice_ok[sli]
        #ffout = os.path.join('/tmp/test_training_ds_python_heading_fix.nc')
        #ffout = os.path.join(OUTPUTDIR,os.path.basename(ff))
    else :
        #ffout = os.path.join(OUTPUTDIR,os.path.basename(ff))
        #N_time = len(nci.dimensions['time'])
        N_time = len(dates_indice_ok)
        sli = dates_indice_ok
        logging.info('N_time : %s',N_time)
    nco = netCDF4.Dataset(ffout,'w')


    #     logging.info('%s/%s %s taille file = %s',ii+1,len(listing_justin),cpt,N_time)
    # creates dimensions
    for di in nci.dimensions :
        logging.debug('%s %s',di,type(di))
        if di == 'time' :
            val_dim = N_time
        elif di == 'num_char' :
            val_dim = 254
        else :
            val_dim = len(nci.dimensions[di])
        nco.createDimension(di,val_dim)

        logging.debug('%s %s %s',di,di,len(di))


    # find original ocn files to read track angle and incidence angle
    ori = {}
    #for xx in range(N_time) :
    for xx in dates_indice_ok:
        ori_path = L2_paths_datarmor[xx]
        logging.debug('ori path: %s',ori_path)
        ncori = netCDF4.Dataset(ori_path)
        for vv in ['oswHeading',
                   'oswIncidenceAngle','oswLon','oswLat'] :  # the 2 variables in ushort un cwaveV4 that need to be replaced (in particular heading)
            if vv == 'oswHeading' :
                if 'trackAngle' not in ori :
                    ori['trackAngle'] = [ncori.variables[vv][0][0]]
                else :
                    ori['trackAngle'].append(ncori.variables[vv][0][0])
            elif vv == 'oswIncidenceAngle' :
                if 'incidenceAngle' not in ori :
                    ori['incidenceAngle'] = [ncori.variables[vv][0][0]]
                else :
                    ori['incidenceAngle'].append(ncori.variables[vv][0][0])
            else:
                if vv not in ori :
                    ori[vv] = [ncori.variables[vv][0][0]]
                else :
                    ori[vv].append(ncori.variables[vv][0][0])
        ncori.close()
    logging.info('end of OCN copy')
    lons = np.array(ori['oswLon'])
    lats = np.array(ori['oswLat'])
    ths1 = np.arange(0,360,5)
    # add the var:

    list_vars = list([uu for uu in nci.variables.keys()])
    if 'th' not in list_vars:
        list_vars += ['th']
    logging.info('list_vars : %s',list_vars)
    for va in list_vars :
        logging.info('vv: %s',va)
        if va != 'S' :
            va = str(va)
            if va in ['th']:
                dims = ('directions')
                tmpva_iin = None
            else:
                dims = nci.variables[va].dimensions
                tmpva_iin = nci.variables[va]
            fv = None

            if va in ['trackAngle']:
                input_values = np.array(ori['trackAngle'])
            elif va in ['incidenceAngle']:
                input_values = np.array(ori['incidenceAngle'])
            elif va in ['lonSAR']:
                input_values = lons
                ty = 'double'
            elif va in ['latSAR']:
                input_values = lats
                ty = 'double'
            elif va in ['th']:
                input_values = ths1 # to avoid issues with values X.00001
            elif va in ['fileNameFull']:  # L2 paths
                input_values = L2_paths_datarmor
            elif va in ['filterNameFull'] : #L1 paths
                tmp_values = []
                for iiu in range(tmpva_iin[:].shape[0]) :
                    apath = ('').join([ddc.decode() for ddc in tmpva_iin[iiu,:]])
                    tmp_values.append(apath.replace('cercache','datawork-cersat-public'))
                input_values = np.array(tmp_values)
            else:
                input_values = copy.copy(tmpva_iin[:])
            # remove the scale factor in output file
            if va not in ['th']:
                if 'scale_factor' in nci.variables[va].ncattrs() :
                    ty = 'double'
                    logging.debug('%s type %s -> %s',va,nci.variables[va].dtype,ty)
                    # input_values = input_values*float(tmpva_iin.getncattr('scale_factor')) #finally it is not necessary to apply it it is already done
                else :
                    ty = nci.variables[va].dtype
            else:
                ty = 'float'

            # replace cercache by datawork
            if va in ['filterNameFull'] : #L1 paths
                dims = ('time')
                tmpva = nco.createVariable(va,'S243',dims,fill_value=fv)
            elif va in ['fileNameFull']: #L2 paths
                dims = ('time')
                tmpva = nco.createVariable(va,'S254',dims,fill_value=fv)
            else :
                tmpva = nco.createVariable(va,ty,dims,fill_value=fv)

            if va not in ['th'] :
                for attr in nci.variables[va].ncattrs() :
                    if attr not in ['_FillValue','scale_factor','add_offset'] :
                        src_att = tmpva_iin.getncattr(attr)
                        tmpva.setncattr(attr,src_att)
            sha = input_values.shape

            if 'time' in dims :

                if len(sha) == 1 :
                    if va in ['latSAR','lonSAR','incidenceAngle','trackAngle']:
                        tmpva[:] = input_values #values already subseted
                    else:
                        tmpva[:] = input_values[sli]
                elif len(sha) == 2 :
                    tmpva[:] = input_values[sli,:]
                else :
                    tmpva[:] = input_values[sli,:]
            else :
                tmpva[:] = input_values[:]
    # add the python S param and the cartesian cross spectra coming from scipy.interpolate.griddata('linear')
    ####################################################
    # if 'th' not in nci.variables:
    #     tmpf = '/home/cercache/project/mpc-sentinel1/data/esa/sentinel-1a/L2/WV/S1A_WV_OCN__2S/2019/249/S1A_WV_OCN__2SSV_20190906T095937_20190906T100447_028899_0346AC_75DC.SAFE/measurement/s1a-wv1-ocn-vv-20190906t100430-20190906t100433-028899-0346ac-021.nc'
    #     nctmp = netCDF4.Dataset(tmpf)
    #     ths1 = nctmp.variables['oswPhi'][:].squeeze()
    #     nctmp.close()
    # else:
    #     ths1 = nci.variables['th'][:]
    # if ths1.mask.any():
    #     logging.info('I correct the theta vector')
    logging.info('variables needed for Cwave computation added')
    dates = nci.variables['timeSAR'][:]
    cspcRes = nci.variables['cspcRe'][:]
    cspcIms = nci.variables['cspcIm'][:]
    dates_dt = netCDF4.num2date(dates,nci.variables['timeSAR'].units)
    ks1 = nci.variables['k'][:]
    # the patch on oswK wave number vector is useless since the dataset provided by J.Stopa has already normalized k vector.
    # if (np.isfinite(ks1) is False).all() or (ks1.mask is True).any():
    # hard choice to do here because, some of the spectra in 2015/05 are described on the grid 60 pts up to 954m but the ohter are
    # on the grid 60 pts up to 1145m. Here we are doing better than what was provided as training dataset for Quach since we prevent to have K vector with masked values
    ks1 = patch_oswK(ks1,ipfvesion=None,datedtsar=dates_dt[0])
    #         logging.debug('no valid wave number value %s',ks1)
    #         if dates_dt[-1]<datetime.datetime(2016,1,1):
    # #             ks1 =reference_oswK_2015 #30 values is not enough even for 2015 S1A data.... i dont know why... may be justin normalized the cross spectra shape..
    #             ks1 = reference_oswK_2017
    #         elif  dates_dt[-1]<datetime.datetime(2018,1,1) and dates_dt[-1]>=datetime.datetime(2016,1,1):
    #             ks1 = reference_oswK_2017
    #         else:
    #             ks1 = reference_oswK_2018
    #         logging.debug('default k is length= %s cspc %s',len(ks1),cspcRes.shape)

    #lons = nci.variables['lonSAR'][:] #replaced by original values in con wv files since there was an 0.002 deg diff not explicated
    #lats = nci.variables['latSAR'][:]

    satellite = os.path.basename(ff)[0 :3].lower()

    #tas = nci.variables['trackAngle'][:]
    #incs = nci.variables['incidenceAngle'][:]
    tas = np.array(ori['trackAngle'])
    incs = np.array(ori['incidenceAngle'])

    # if True: #test to replace the incidence angle and trackangle to see if the difference in S params is lower (Feb 2021)

    nvs = nci.variables['normalizedVariance'][:]
    sig0s = nci.variables['sigma0'][:]
    #for jj in range(N_time) :
    logging.info('start CWAVE computation on %s obs',len(dates_indice_ok))
    for jx,jj in enumerate(dates_indice_ok):
        cspcRe = cspcRes[jj,:]
        cspcIm = cspcIms[jj,:]
        ta = tas[jx]
        incidenceangle = incs[jx]
        nv = nvs[jj]
        s0 = sig0s[jj]
        datedt = dates_dt[jj]
        lonsar = lons[jx]
        latsar = lats[jx]
        # if jj==0: #save a pickle for debug/test
        #     import pickle
        #     savings = {'cspcRe':cspcRe,'cspcIm':cspcIm,'ta':ta,'incidenceangle':incidenceangle,'nv':nv,'s0':s0,'datedt':datedt,
        #                'lonsar':lonsar,'latsar':latsar}
        #     outputpl = '/tmp/hs_sar_training_dataset_vars_before_cwave_compute_%s.pkl'%(datedt.strftime('%Y%m%dT%H%M%S'))
        #     fifi = open(outputpl,'wb')
        #     pickle.dump(savings,fifi)
        #     fifi.close()
        #     logging.info('pickle: %s',outputpl)

        #             ths1,ks1,ta,incidenceangle,s0,nv,cspcRe,cspcIm,datedt,lonsar,latsar,satellite
        subset_ok,flagKcorrupted,cspcReX,cspcImX,cspcRev2,ks1,ths1,kx,ky,cspcReX_not_conservativ,S = format_input_CWAVE_vector_from_OCN(
            cspcRe=cspcRe.T,cspcIm=cspcIm.T,ths1=ths1,ta=ta,incidenceangle=incidenceangle,s0=s0,nv=nv,ks1=ks1,
            datedt=datedt,lonSAR=lonsar,latSAR=latsar,satellite=satellite)
        if jx == 0 :
            Spy = S
            big_Re_cart = cspcReX
            big_Rim_cart = cspcImX
            big_flag_usable = flagKcorrupted
        else :
            Spy = np.hstack([Spy,S])
            big_Re_cart = np.dstack([cspcReX,big_Re_cart])
            big_Rim_cart = np.dstack([cspcImX,big_Rim_cart])
            big_flag_usable = np.hstack([big_flag_usable,flagKcorrupted])
            #                 big_Rim_cart = np.concatenate([cspcImX,big_Rim_cart],axis=2)
            if jx % 50 == 0 :
                #logging.info('%s/%s',jx,len(nci.dimensions['time']))
                logging.info('%s/%s',jx,N_time)
                logging.info('taille big_Rim_cart %s',big_Rim_cart.shape)
                logging.info('taille S param %s',Spy.shape)
    logging.info('end of the CWAVE computation %s',N_time)
    logging.info('rool axis before %s',big_Re_cart.shape)
    big_Re_cart = np.rollaxis(big_Re_cart,axis=2,start=0)
    big_Rim_cart = np.rollaxis(big_Rim_cart,axis=2,start=0)
    logging.info('rool axis after %s',big_Re_cart.shape)
    nco.createDimension('cartX',big_Re_cart.shape[1])
    nco.createDimension('cartY',big_Re_cart.shape[2])
    cartRevar = nco.createVariable('py_cspcReX','f8',('time','cartX','cartY'))
    cartRevar.setncattr('long_name',
                        'Real part of cartesian wave height spectra coming from Python interpolation using griddata scipy %s' % (
                            sciV))
    cartImvar = nco.createVariable('py_cspcImX','f8',('time','cartX','cartY'))
    cartImvar.setncattr('long_name',
                        'Imaginary part of cartesian wave height spectra coming from Python interpolation using griddata scipy %s' % (
                            sciV))
    kxvar = nco.createVariable('py_kx','f8',('cartX'))
    kxvar.setncattr('long_name','kx to go with py_cspcImX and py_cspcReX')
    kyvar = nco.createVariable('py_ky','f8',('cartY'))
    kyvar.setncattr('long_name','ky to go with py_cspcImX and py_cspcReX')
    kxvar[:] = kx
    kyvar[:] = ky
    logging.info('fill %s with %s',cartRevar,big_Re_cart.shape)
    cartRevar[:] = big_Re_cart
    cartImvar[:] = big_Rim_cart
    pySvar = nco.createVariable('py_S','f8',('time','N'))
    Spy = np.rollaxis(Spy,axis=1)
    logging.info('spy %s %s',pySvar,Spy.shape)
    str_long = 'CWAVE S 20 parameters from python %s numpy %s scipy %s' % (pyv,numV,sciV)
    logging.info('str_long = %s %s',str_long,type(str_long))
    try :
        pySvar.setncattr('long_name',str_long)
    except :
        pass

    flagvar = nco.createVariable('flag_usable_acquisition','int',('time'))
    flagvar[:] = (big_flag_usable == False)
    flagvar.setncattr('descr','0 = not usable, 1 = usable for training')
    flagvar.setncattr('long_name','flag taking into account wavenumber vector corrupted or empty input cross spectra')

    pySvar[:] = Spy
    #add prediction Hs and Hs STD
    #spectrum = np.stack((big_Re_cart, big_Rim_cart), axis=3)
    #logging.info('spectrum shape : %s',spectrum.shape)
    #ds_training = xarray.Dataset()
    #the vars I need: ['cwave','dxdt','latlonSARcossin','todSAR','incidence','satellite']
    #ds_training['S'] = Spy
    #ds_training['Re'] = big_Re_cart
    #ds_training['Im'] = big_Rim_cart
    #features = define_features(ds_training)


    nci.close()
    nco.close()
    logging.info('rewrite %s',ff)
    logging.info('output tmp %s exist? %s',ffout,os.path.exists(ffout))
    time.sleep(1)
    dsout = xarray.open_dataset(ffout)
    logging.info('peak memory usage: %s kilobytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return dsout,ffout

#def get_input_output(inputfile,model,dev=False):
def get_input_output ( satellite,datedaydt,model,dev=False ,redo=False) :
    """
    :param satellite: str
    :param datedaydt: datetime
    :param inputfile: cwave V6 nc files (monthly)
    :param model:
    :param dev:
    :return:
    """
    ffout = os.path.join(OUTPUTDIR_TMP,
                         '%s_%s_ifr_tmp_input_output_quach2020_python.nc' % (satellite,datedaydt.strftime('%Y%m%d')))
    #dirname_inp = os.path.dirname(ffout)
    #dirname_out = os.path.join(dirname_inp,'final')
    final_output = os.path.join(OUTPUTDIR_FINAL,os.path.basename(ffout).replace('.nc','v2.nc'))
    if os.path.exists(final_output) and redo==False:
        logging.info('output final %s already exists and redo = False',final_output)
    else:
        ds_train_raw,ffpathout = append_python_vars_ds2019(satellite,datedaydt,dev=dev,ffout=ffout)
        #ds_train_raw,ffpathout = append_python_vars_ds2019(inputfile,dev)
        spectrum_x,ds_training_normalized_x = prepare_training_dataset_core(ds_train_raw,validation_dataset=True)
        ds_training_with_prediciton = compute_prediction_from_training_inputs(ds_training_normalized_x,spectrum_x,model)

        ds_training_with_prediciton.to_netcdf(final_output)
        logging.info('final output : %s',final_output)

if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser(description='produce input ouput validation dataset Hs NN SAR')
    parser.add_argument('--verbose',action='store_true',default=False)
    parser.add_argument('--redo',action='store_true',default=False,
                        help='activate overwritting of existing output netCDF default is False (ie keep existing)',required=False)
    parser.add_argument('--dev',action='store_true',default=False,
                        help='just for test to not erase exiting file (smallest amount of acquisition = 10)')
    # parser.add_argument('--redo', action='store_true',default=False,help='redo already having fileNameFull var files ')
    #parser.add_argument('--input',action='store',required=False,default=None,help='input nc file to treat ')
    parser.add_argument('--date',action='store',required=True,default=None,help='date YYYYMMDD to treat ')
    parser.add_argument('--satellite',help='S1A or S1B',required=True)
    args = parser.parse_args()
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                            datefmt='%d/%m/%Y %I:%M:%S')
    heteroskedastic_2017 = load_quach2020_model_v2()
    dateday_dt = datetime.datetime.strptime(args.date,'%Y%m%d')
    #if args.input is None :
    #    raise Exception()
        # find the files from ju:stin (coloc with alti)


        #listing_justin = glob.glob(os.path.join(input_justin,'S1*_2019*S.nc'))
        #logging.info('Nb files justin : %s',len(listing_justin))
        #cpt = collections.defaultdict(int)
        #for ii,ff in enumerate(listing_justin) :
        #    logging.info('input: %s',ff)
        #    get_input_output(inputfile=ff,model=heteroskedastic_2017,dev=args.dev)
        #    break
        #logging.info('cpt%s',cpt)
    #else :
    #get_input_output(da=args.input,model=heteroskedastic_2017,dev=args.dev)
    get_input_output(args.satellite,dateday_dt,model=heteroskedastic_2017,dev=args.dev,redo=args.redo)
    print('end in append...py')