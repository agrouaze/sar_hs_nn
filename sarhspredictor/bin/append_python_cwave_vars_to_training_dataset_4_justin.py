"""
context: improvment hs total SAR
creation date: 10 sept 2019
author Antoine Grouazel
prupose: add 20 python CWAVE params to existing training CWAVE dataset in order to retrain the NN model having complaint inputs with IPF

copy/paste from /home1/datahome/agrouaze/sources/sentinel1/hs_total/append_python_cwave_vars_to_training_dataset_4_justin.py

for gogolist run on the 65 inputs files:
listing = /home/cercache/users/agrouaze/gogolist/cwave_apend_py_var_2_training_dataset_v2.txt
env : source /home/losafe/users/agrouaze/conda/miniconda3/bin/activate /home/losafe/users/agrouaze/conda/miniconda3/envs/agrouaze_env2.7/
exe : /home/agrouaze/workspace_leste/sentinel1_perso/append_python_cwave_vars_to_training_dataset_4_justin.py

note: cwaveV4 dataset contains spectra with dimensions 72x60 whatever the version of IPF of the original product, products (for instance in 2015 from Jan to May)
 with dimensions different than 72x60 are filled with zero matrix.
 K (wave number vector) is  changing from a monthly file to another following evolution of the ESA products.
 in 201507 for instance the file contains masked values, thus a patch is necessary

update January 2021: preparation of traning dataset for Quach 2020 model rebuild
env :/home1/datawork/agrouaze/conda_envs2/envs/cwave/bin/python (env used to provided the requirement_light.txt file)

"""
import os
import glob
import logging
import netCDF4
import scipy
import collections
import copy
import numpy as np
import pdb
import sys
import datetime
from sarhspredictor.lib.apply_oswK_patch import patch_oswK
from sarhspredictor.lib.compute_CWAVE_params import format_input_CWAVE_vector_from_OCN
sciV = scipy.__version__
numV=np.__version__
pyv=sys.version_info
#sys.path.append('/home/agrouaze/git/npCWAVE')
#sys.path.append('/home/agrouaze/git/mpc-sentinel/mpc-sentinel/mpcsentinellibs/data_collect/')
#sys.path.append('/home/agrouaze/git/mpc-sentinel/mpc-sentinel/mpcsentinellibs/qualitycheck/')
#sys.path.append('/home/agrouaze/git/mpc-sentinel/mpc-sentinel/mpcsentinellibs/colocation/ww3spectra')
output = '/home/cercache/users/jstopa/sar/empHs/cwaveV5'
output = '/home1/scratch/agrouaze/cwaveV4_python'
output = '/home1/datawork/agrouaze/data/sentinel1/cwave/training_dataset_quach2020_python'
#from compute_hs_total_SAR_v2 import prepare_input_CWAVE_vector_from_OCN,format_input_CWAVE_vector_from_OCN
#from reference_oswk import reference_oswK_2015,reference_oswK_2017,reference_oswK_2018
# from generate_track_file_from_SLC import input_track_names,REP_COLOC
#from get_full_path_from_measurement import get_full_path_from_measu
def mock_full_path():
    tmp = np.zeros(240,dtype=str)
    tmp[:] = 'e'
    tmp = list(tmp)
    res = ''.join(tmp)
    return res

def append_python_vars(ff,dev=False):
    """

    :param ff:
    :param dev:
    :return:
    """
    logging.info('treat %s',ff)
    
        
    nci = netCDF4.Dataset(ff,'r')
    
    if dev is True:
        N_time = 10
        ffout = os.path.join('/tmp/test_cwavev2python.nc')
    else:
        ffout = os.path.join(output,os.path.basename(ff))
        N_time = len(nci.dimensions['time'])
    nco = netCDF4.Dataset(ffout,'w')
    logging.warning('for dev N_time = %s',N_time)

#     logging.info('%s/%s %s taille file = %s',ii+1,len(listing_justin),cpt,N_time)
    #creates dimensions
    for di in nci.dimensions:
        logging.debug('%s %s',di,type(di))
        if di=='time':
            val_dim = N_time
        elif di=='num_char':
            val_dim = 254
        else:
            val_dim = len(nci.dimensions[di])
        nco.createDimension(di,val_dim)
        
        logging.debug('%s %s %s',di,di,len(di))
    #add the var:
    sli = slice(0,N_time)
    for va  in nci.variables.keys():
        if va != 'S':
            va = str(va)
            dims = nci.variables[va].dimensions
            fv = None
            tmpva_iin = nci.variables[va]
            input_values = copy.copy(tmpva_iin[:])
            # remove the scale factor in output file
            if 'scale_factor' in nci.variables[va].ncattrs():
                ty = 'f8'
                logging.debug('%s type %s -> %s',va,nci.variables[va].dtype,ty)
                #input_values = input_values*float(tmpva_iin.getncattr('scale_factor')) #finally it is not necessary to apply it it is already done
            else:
                ty = nci.variables[va].dtype
            # replace cercache by datawork
            if va in ['fileNameFull','filterNameFull'] :
                tmp_values = []
                for iiu in range(input_values.shape[0]) :
                    # print('raw',input_values[iiu,:])
                    apath = ('').join([ddc.decode() for ddc in input_values[iiu,:]])
                    # print(va,apath)
                    tmp_values.append(apath.replace('cercache','datawork-cersat-public'))
                input_values = np.array(tmp_values)[sli]
                #if va=='fileNameFull':
                dims = ('time')
                if va=='fileNameFull':
                    tmpva = nco.createVariable(va,'S254',dims,fill_value=fv)
                else:
                    tmpva = nco.createVariable(va,'S243',dims,fill_value=fv)
            else:
                tmpva = nco.createVariable(va,ty,dims,fill_value=fv)
            
            for attr in nci.variables[va].ncattrs():
                if attr not in ['_FillValue','scale_factor','add_offset']:
                    src_att = tmpva_iin.getncattr(attr)
                    tmpva.setncattr(attr,src_att)
            sha = input_values.shape


            if 'time' in dims:
                
                
                if len(sha)==1:
                    tmpva[:] = input_values[sli]
                elif len(sha)==2:
                    tmpva[:] = input_values[sli,:]
                else:
                    tmpva[:] = input_values[sli,:]
            else:
                tmpva[:] = input_values[:]
    #add the python S param and the cartesian cross spectra coming from scipy.interpolate.griddata('linear')
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
    ths1 = np.arange(0,360,5)
    dates = nci.variables['timeSAR'][:]
    cspcRes = nci.variables['cspcRe'][:]
    cspcIms = nci.variables['cspcIm'][:]
    dates_dt = netCDF4.num2date(dates,nci.variables['timeSAR'].units)
    ks1 = nci.variables['k'][:]
    #the patch on oswK wave number vector is useless since the dataset provided by J.Stopa has already normalized k vector.
    #if (np.isfinite(ks1) is False).all() or (ks1.mask is True).any():
    # hard choice to do here because, some of the spectra in 2015/05 are described on the grid 60 pts up to 954m but the ohter are
    #on the grid 60 pts up to 1145m. Here we are doing better than what was provided as training dataset for Quach since we prevent to have K vector with masked values
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
    
    lons = nci.variables['lonSAR'][:]
    lats = nci.variables['latSAR'][:]
    
    satellite = os.path.basename(ff)[0:3]

    tas = nci.variables['trackAngle'][:]
    incs = nci.variables['incidenceAngle'][:]
    #if True: #test to replace the incidence angle and trackangle to see if the difference in S params is lower (Feb 2021)


    nvs = nci.variables['normalizedVariance'][:]
    sig0s = nci.variables['sigma0'][:]
    for jj in range(N_time):
        cspcRe = cspcRes[jj,:]
        cspcIm = cspcIms[jj,:]
        ta = tas[jj]
        incidenceangle =  incs[jj]
        nv = nvs[jj]
        s0 = sig0s[jj]
        datedt = dates_dt[jj]
        lonsar = lons[jj]
        latsar = lats[jj]
#             ths1,ks1,ta,incidenceangle,s0,nv,cspcRe,cspcIm,datedt,lonsar,latsar,satellite 
        subset_ok,flagKcorrupted,cspcReX,cspcImX,cspcRev2,ks1,ths1,kx,ky,cspcReX_not_conservativ,S = format_input_CWAVE_vector_from_OCN(
            cspcRe=cspcRe.T,cspcIm=cspcIm.T,ths1=ths1,ta=ta,incidenceangle=incidenceangle,s0=s0,nv=nv,ks1=ks1,
            datedt=datedt,lonSAR=lonsar,latSAR=latsar,satellite=satellite)
        if jj==0:
            Spy = S
            big_Re_cart = cspcReX
            big_Rim_cart = cspcImX
            big_flag_usable = flagKcorrupted
        else:
            Spy = np.hstack([Spy,S])
            big_Re_cart = np.dstack([cspcReX,big_Re_cart])
            big_Rim_cart = np.dstack([cspcImX,big_Rim_cart])
            big_flag_usable = np.hstack([big_flag_usable,flagKcorrupted])
#                 big_Rim_cart = np.concatenate([cspcImX,big_Rim_cart],axis=2)
            if jj%50==0:
                logging.info('%s/%s',jj,len(nci.dimensions['time']))
                logging.info('taille big_Rim_cart %s',big_Rim_cart.shape)
                logging.info('taill S param %s',Spy.shape)
    logging.info('rool axis before %s',big_Re_cart.shape)
    big_Re_cart = np.rollaxis(big_Re_cart,axis=2,start=0)
    big_Rim_cart = np.rollaxis(big_Rim_cart,axis=2,start=0)
    logging.info('rool axis after %s',big_Re_cart.shape)
    nco.createDimension('cartX',big_Re_cart.shape[1])
    nco.createDimension('cartY',big_Re_cart.shape[2])
    cartRevar = nco.createVariable('py_cspcReX','f8',('time','cartX','cartY'))
    cartRevar.setncattr('long_name','Real part of cartesian wave height spectra coming from Python interpolation using griddata scipy %s'%(sciV))
    cartImvar = nco.createVariable('py_cspcImX','f8',('time','cartX','cartY'))
    cartImvar.setncattr('long_name','Imaginary part of cartesian wave height spectra coming from Python interpolation using griddata scipy %s'%(sciV))

    cartRevar[:] = big_Re_cart
    cartImvar[:] = big_Rim_cart
    pySvar = nco.createVariable('py_S','f8',('time','N'))
    Spy = np.rollaxis(Spy,axis=1)
    logging.info('spy %s %s',pySvar,Spy.shape)
    str_long = 'CWAVE S 20 parameters from python %s numpy %s scipy %s'%(pyv,numV,sciV)
    logging.info('str_long = %s %s',str_long,type(str_long))
    try:
        pySvar.setncattr('long_name',str_long)
    except:
        pass
    
    flagvar = nco.createVariable('flag_usable_acquisition','int',('time'))
    flagvar[:] = (big_flag_usable==False)
    flagvar.setncattr('descr','0 = not usable, 1 = usable for training')
    flagvar.setncattr('long_name','flag taking into account wavenumber vector corrupted or empty input cross spectra')
    
    
    pySvar[:] = Spy
    nci.close()
    nco.close()
    logging.info('rewrite %s',ff)
    logging.info('output %s',ffout)
if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='my title')
    parser.add_argument('--verbose', action='store_true',default=False)
    parser.add_argument('--dev', action='store_true',default=False,help='just for test to not erase exiting file (smallest amount of acquisition = 10)')
    #parser.add_argument('--redo', action='store_true',default=False,help='redo already having fileNameFull var files ')
    parser.add_argument('--input', action='store',required=False,default=None,help='input nc file to treat ')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S')
    if args.input is None:
        #find the files from ju:stin (coloc with alti)
        input_justin = '/home/cercache/users/jstopa/sar/empHs/cwaveV4'
        
        listing_justin = glob.glob(os.path.join(input_justin,'*.nc'))
        logging.info('Nb files justin : %s',len(listing_justin))
        cpt = collections.defaultdict(int)
        for ii,ff in enumerate(listing_justin):
            append_python_vars(ff,dev=args.dev)
            break
        logging.info('cpt%s',cpt)
    else:
        print('treat ',args.input)
        append_python_vars(args.input,args.dev)
        print('end in append...py')
#         break
    #find initial trackfile
    # dirout = os.path.join(REP_COLOC[satsat],year+month+'/')
    # input_track_names(satellite,year,month,level,dirout)
    
    #add the full path /home/cercache