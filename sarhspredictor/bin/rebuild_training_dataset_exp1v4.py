"""
exp 1: doing the Quach experiment with X spectra from SLC instead of OCN
 steps
 1) read a tiff
 2) find the indice of this tiff in the cwaveV4 to retrieve Hs alti
 4) add SLC cross spectra
 (note that I could keep the CWAVE params from L2 xspectra... not tested yet)
Oct 2021
A Grouazel
the listing of tiff to treat is here: /home1/datawork/agrouaze/data/sentinel1/cwave/listing_SAR_L2_L1_measu_from_colocations_cwaveV4.txt
-no CWAVE
-no wv_angle
-replaced hour of day by day of year
-no more OCN inputs
-incidence angle normalized for IW range
INPUTS: SLC WV, CWAVEv4 alti Hs (justin),
env : xsar_gdal3.3_v2 -> puis xsar_pr46 en dev/test Juin 2022
NOTE: ce script ne sera peut etre jamais utilise car je peux repartir du dataset D1_v3, il n'y a que la normalization
et les output varibles qui changent
"""
import pdb

import os
import sys
sys.path.append('/home1/datahome/agrouaze/git/mpc/data_collect')
sys.path.append('/home1/datahome/agrouaze/git/xsarseafork/src/xsarsea')
sys.path.append('/home1/datahome/agrouaze/git/xsarseafork/src/')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/lib')
import logging
import xarray
import numpy as np
import datetime
import xsar
import glob
import warnings
import time
from collections import defaultdict
import resource
warnings.simplefilter(action='ignore',category=FutureWarning)
from scipy.spatial import KDTree

from predict_hs_from_SLC import get_xspectrum_SLC,reference_oswK_1145m_60pts,compute_Cwave_params_and_xspectra_fromSLC
from compute_CWAVE_params import format_input_CWAVE_vector_from_OCN
from slc_image_normalization import gaussian_lowpass

DIR_ORIGINAL_COLOCS = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/quach2020/validation/colocations/original_colocations_YoungAltiDatabase_vs_WV_L2_Jstopa/cwaveV4/'
OUTPUTDIR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v1/' # D1v4 10oct21
OUTPUTDIR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v6/' # 8 may 2022 completed 17 may 22
OUTPUTDIR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v7/' # 20 may 2022 rotation fixed en -90° pour asc&desc, no more NaN in spectrum, no more abs(Imag)
OUTPUTDIR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1v4/training_dataset/v8/' # 11 Juin 2022: dataset avec tte les variable des hawaiiens HLF + cwave + spectre
def round_seconds(obj: datetime.datetime) -> datetime.datetime:
    if obj.microsecond >= 500_000:
        obj += datetime.timedelta(seconds=1)
    return obj.replace(microsecond=0)


def find_indices_sar_acquisition_in_monthly_colocs(ds_coloc,datedt_ocn,lon_x,lat_x,new_times):
    """ds_coloc
    knowing the date SAR seek I want the indice(s) in Justin's dataset (many cases possibles)
    debug in debug_method_to_associate_SAR_measu_with_ALTI_from_monthly_cwaveV4_JS_dataset.ipynb
    :param ds_coloc:
    :param datedt_ocn:
    :param lon_x:
    :param lat_x:
    :return:
    """
    cpt = defaultdict(int)
    logging.debug('after int cast : %s',ds_coloc['timeSAR'])
    # logging.debug('after decode_cf %s',ds_coloc['timeSAR'])
    # 0) find the proper indice of the ocn file in coloc file
    # logging.debug('timeSAR val %s',ds_coloc['timeSAR'].values)
    logging.debug('timeSAR %s',ds_coloc['timeSAR'])
    # ind = np.where(ds_coloc['fileNameFull']==os.path.basename(ocn))
    inds = np.where(new_times == datedt_ocn)[0]
    if inds.size == 0 :
        tmp_inds = np.where(abs(new_times - datedt_ocn) == np.amin(abs(new_times - datedt_ocn)))
        logging.debug('tmp_inds  :%s %s while seek %s',tmp_inds,new_times[tmp_inds],datedt_ocn)
        if len(tmp_inds[0]) > 1  and datedt_ocn.year==2015: # special case where the all the measurement ocn had hte same start dates (in filename )
            cpt['many_closest_in2015'] += 1
            tmpllons = []
            tmplats = []
            for yy in range(len(tmp_inds[0])) :
                logging.debug('seek %s %s %s ,dates_parsed[closest %s lon: %s lat : %s',datedt_ocn,lon_x,lat_x,
                             new_times[tmp_inds[0][yy]],
                             ds_coloc['lonSAR'].values[tmp_inds[0][yy]],
                             ds_coloc['latSAR'].values[tmp_inds[0][yy]],
                             )
                tmpllons.append(ds_coloc['lonSAR'].values[tmp_inds[0][yy]])
                tmplats.append(ds_coloc['latSAR'].values[tmp_inds[0][yy]])
            tree = KDTree(np.c_[tmpllons,tmplats])
            dd,ii = tree.query([lon_x,lat_x],k=1)
            if dd<2: #je veerifie qu il  aune distance inference a 2 degrees
                logging.info('dd %s ii : %s',dd,ii)
                inds = [tmp_inds[0][ii]]  # to get the good measurement
            else:
                inds = []
        elif len(tmp_inds[0]) > 1  and datedt_ocn.year!=2015:
            # je considere que si on est apres 2015, tous les doublons sont des acquisitions SAR avec match sur plusieurs alti
            # -> je garde tous les indices
            inds = tmp_inds
            cpt['many_closest_after2015']+=1
        else :
            cpt['unique_closest'] += 1
            inds = tmp_inds[0]
    else :
        logging.debug('direct match')
        cpt['direct_indice_matching'] += 1

    logging.debug('vars : %s',[qq for qq in ds_coloc.keys() if 'hs' in qq])
    logging.debug('ind: %s how it has been found : %s',inds,
                 cpt)  # it could be that there are many indices (same SAR acquisition mathcing different alti points)
    for ind in inds:# small check on the hs value from altimeter
        logging.debug('Hs alti : %1.4fm',ds_coloc['hsALT'].isel({'time':ind}))
    return inds,ds_coloc,cpt

def read_coloc_file(slc,ocn):
    """
    for one pair tiff ocn -> I return the geoloc (OCN) the date (SLC) the dataset containing this pair in Justin dataset
     and the times rounded from this last dataset
    :param slc:
    :param ocn:
    :return:
        new_times : nb array of datetime obj SAR dates without microseconds
    """
    datedt_slc = datetime.datetime.strptime(os.path.basename(slc).split('-')[4],'%Y%m%dt%H%M%S')
    dsocn = xarray.open_dataset(ocn)
    lon_x = dsocn['oswLon'].values.squeeze()
    lat_x = dsocn['oswLat'].values.squeeze()
    sat = os.path.basename(slc)[0 :3]
    coloc_file_JS = os.path.join(DIR_ORIGINAL_COLOCS,
                                 sat.upper() + '_ALT_coloc' + datedt_slc.strftime('%Y%m') + 'S.nc')
    logging.debug('coloc_file_JS : %s',coloc_file_JS)
    ds_coloc = xarray.open_dataset(coloc_file_JS)
    ds_coloc['lonSAR'] = ds_coloc['lonSAR'].persist()
    ds_coloc['latSAR'] = ds_coloc['latSAR'].persist()
    new_times = []
    for oo in range(len(ds_coloc['timeSAR'])) :
        tmpvalt = ds_coloc['timeSAR'].values[oo]
        ts = (tmpvalt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
        dt3 = datetime.datetime.utcfromtimestamp(ts)
        tmpvalt2 = round_seconds(dt3)
        new_times.append(tmpvalt2)
    new_times = np.array(new_times)
    return new_times,lon_x,lat_x,sat,ds_coloc,datedt_slc



def prepare_one_measurement(slc,ocn,dev=False,add_cwave_L2=False,add_cwave_L1=False):
    """

    :param slc: str
    :param ocn : str I still need the ocn path since it is the key for Justins' coloc
    :param coloc_file_JS: str full path of the colocation file from Justin Stopa
    :return:
    """
    logging.debug('slc :%s',slc)
    logging.debug('ocn: %s',ocn)
    # read SLC tiff files with xsar
    imagette_number = os.path.basename(slc).split('-')[-1].replace('.tiff', '')
    print(imagette_number)
    fullpathsafeSLC = os.path.dirname(os.path.dirname(slc))
    if 'WV' in slc:
        str_gdal = 'SENTINEL1_DS:%s:WV_%s' % (fullpathsafeSLC, imagette_number)
    print('str_gdal', str_gdal)
    xsarobj = xsar.Sentinel1Dataset(str_gdal)


    new_times,lon_x,lat_x,sat,ds_coloc,datedt_ocn = read_coloc_file(slc,ocn)
    inds,ds_coloc,cptx = find_indices_sar_acquisition_in_monthly_colocs(ds_coloc,datedt_ocn,lon_x,lat_x,new_times)
    vars_2_drop = ['dx','dt','altID','hsSM','S','fileNameFull','filterNameFull','cspcRe','cspcIm']
    vars_2_drop = ['S'] #for the dataset needed by Thomas Grange 10 June 2022, I want to put back all the variables, except the CWAVE computed by Justin on matlab spectrum from L2
    ds_coloc = ds_coloc.drop(vars_2_drop)

    nb_match = len(inds)
    all_subsets_coloc = []
    for xxx,indxs in enumerate(inds):
        logging.info('prepare coloc : %s/%s',xxx+1,nb_match)
        # 1) read all params from Justin s dataset
        subsetcoloc = ds_coloc.isel(time=np.array([indxs])) #{ on peut selectionner plusieurs indice en meme temps avec isel}
        subsetcoloc = subsetcoloc.drop('k')
        #subsetcoloc = subsetcoloc.drop('directions')
        logging.info('subsetcoloc : %s',subsetcoloc)
        # keep a time dimension to be sure to be able to gather with open_mfdataset
        for vv in subsetcoloc.keys():
            logging.debug('vv: %s %s',vv,subsetcoloc[vv].shape)
            if vv in ['S','cspcRe','cspcIm']:
                #subsetcoloc = subsetcoloc.drop(vv)
                subsetcoloc[vv].attrs['description'] = 'variable from cwaveV4 matlab interpolation from OCN products'
                # I dont when to drop these variable since I want to do both training for OCN and SLC for exp1 (July 21)
            elif vv in ['phi','k','th']:
                pass
            else:
                pass # finalement ca fout la m..... d avoir une dim time en plus
                #subsetcoloc[vv] = subsetcoloc[vv].assign_coords({'time':[datedt_slc]})
                #if subsetcoloc[vv].size>1:
                # reshped_data = subsetcoloc[vv].values.reshape((1,)+subsetcoloc[vv].shape)
                # new_dims = ['time']+list(subsetcoloc[vv].dims)
                # new_coords = copy.copy(subsetcoloc[vv].coords)
                # logging.info('new_coords : %s %s %s',type(new_coords),new_coords,new_coords.keys())
                # if new_coords.dims==():
                #     new_coords = {'time':[datedt_slc]}
                # else:
                #     new_coords['time'] = [datedt_slc]
                # subsetcoloc[vv] = xarray.DataArray(reshped_data,dims=new_dims,
                #                                    coords=new_coords)
        #change units for timeSAR
        subsetcoloc['timeSAR'].encoding['units'] = 'seconds since 2014-01-01 00:00:00'
        ths1 = np.arange(0, 360, 5)
        s0 = subsetcoloc['sigma0'].values[0] # from L2
        s0_SLC = np.nanmean(xsarobj.dataset['sigma0'].sel(pol='VV').values)
        subsetcoloc['s0_SLC'] = xarray.DataArray([10.*np.log10(s0_SLC)],dims=['time'],attrs={'description':'sigma0 denoised','unit':'dB'})
        nv = subsetcoloc['normalizedVariance'].values[0] # from L2
        intensity = (np.abs(xsarobj.dataset['digital_number'].sel(pol='VV').values) ** 2.).astype('float64')
        lowpass_width = [750., 750.]
        im_spacing = np.array([xsarobj.dataset.attrs['pixel_atrack_m'],xsarobj.dataset.attrs['pixel_xtrack_m']])
        lowpass_sigma = np.array(lowpass_width) / im_spacing
        lowpass = gaussian_lowpass(intensity, lowpass_sigma)
        intensity /= lowpass
        nv_SLC =np.var(intensity) / np.mean(intensity) ** 2.
        subsetcoloc['nv_SLC'] = xarray.DataArray([nv_SLC], dims=['time'],attrs={'description':'normalize variance computed from SLC Digital Numbers'})
        ta_SLC = xsarobj.dataset.platform_heading
        subsetcoloc['ta_SLC'] = xarray.DataArray([ta_SLC], dims=['time'],attrs={'description':'track angle from SLC annotation platform heading'})
        ta = subsetcoloc['trackAngle'].values[0]
        incidenceangle = subsetcoloc['incidenceAngle'].values[0] #from L2
        incidenceangle_SLC = xsarobj.s1meta.image['incidence_angle_mid_swath']
        subsetcoloc['incidenceangle_SLC'] = xarray.DataArray([incidenceangle_SLC], dims=['time'], attrs={
            'description': 'incidence angle at mid swath from SLC annotation '})
        lonsar = subsetcoloc['lonSAR'].values[0]
        latsar = subsetcoloc['latSAR'].values[0]
        lonsar_SLC,latsar_SLC = xsarobj.dataset.footprint.centroid.xy
        subsetcoloc['lonsar_SLC'] = xarray.DataArray([lonsar_SLC[0]], dims=['time'], attrs={
            'description': 'longitude of the SLC footprint centroid'})
        subsetcoloc['latsar_SLC'] = xarray.DataArray([latsar_SLC[0]], dims=['time'], attrs={
            'description': 'latitude of the SLC footprint centroid'})
        if add_cwave_L2:
            #4) compute C wave params
            #subsetcoloc['cspcRe']  = subsetcoloc['cspcRe'].astype('float64').rename({'directions':'phi','wavenumbers':'k'})
            subsetcoloc['cspcRe'] = xarray.DataArray(subsetcoloc['cspcRe'].astype('float64').values,
                                                     coords={'k':reference_oswK_1145m_60pts,'phi':ths1,'time':subsetcoloc['time']},
                                                     dims=['time','phi','k'],attrs={'description':'oswQualityCrossSpectraRe from Justin dataset CWAVEv4',
                                                                                    "long_name" : "SAR: cspcRe real component of the cross spectrum estimated from the SAR image (log-polar grid)"})
            #subsetcoloc['cspcIm']  = subsetcoloc['cspcIm'].astype('float64').rename({'directions':'phi','wavenumbers':'k'}) # not sure this is necessary
            subsetcoloc['cspcIm'] = xarray.DataArray(subsetcoloc['cspcIm'].astype('float64').values,
                                                     coords={'k':reference_oswK_1145m_60pts,'phi':ths1,'time':subsetcoloc['time']},
                                                     dims=['time','phi','k'],attrs={'description':'oswQualityCrossSpectraIm from Justin dataset CWAVEv4',
                                                                                    "long_name" : "SAR: cspcIm imaginary component of the cross spectrum estimated from the SAR image (log-polar grid)"})
            #subsetcoloc['S'] = subsetcoloc['S'].astype('float64') # problem with cwaveV4 variable not readable, I have to compute it again
            satellite = os.path.basename(ocn)[0:3]
            ks1 = reference_oswK_1145m_60pts
            logging.info("subsetcoloc['cspcIm'].values.squeeze().T :%s",subsetcoloc['cspcIm'].values.squeeze().T.shape)
            subset_ok,flagKcorrupted,cspcReX,cspcImX,_,ks1,ths1,kx,ky,\
            cspcReX_not_conservativ,Socn = format_input_CWAVE_vector_from_OCN(cspcRe=subsetcoloc['cspcRe'].values.squeeze().T,
                                                                                cspcIm=subsetcoloc['cspcIm'].values.squeeze().T,
                                                                                ths1=ths1,ta=ta,
                                                                                incidenceangle=incidenceangle,
                                                                                s0=s0,nv=nv,ks1=ks1,datedt=datedt_ocn,
                                                                                lonSAR=lonsar,latSAR=latsar,satellite=satellite)
            subsetcoloc['S'] = xarray.DataArray(np.tile(Socn.T, (1, 1)), dims=['time', 'N'],
                                                coords={'time': subsetcoloc['time'], 'N': np.arange(20)})
            subsetcoloc['S'].attrs['description'] = 'S params from OCN polar xpectra'



        #same operation bu using level1 informations
        #crossSpectraImPol_xa,crossSpectraRePol_xa,crossSpectraImPolval,crossSpectraRePolval,datedt_slc,times_bidons = \
        #    get_xspectrum_SLC(slc,nb_match=nb_match,dev=dev)
        if add_cwave_L1:
            crossSpectraImPol_xa,crossSpectraRePol_xa,times_bidons,S_slc = compute_Cwave_params_and_xspectra_fromSLC(slc,dev,
                             nb_match=1,ths1=ths1,ta=ta,s0=s0,nv=nv,incidenceangle=incidenceangle,lonsar=lonsar,latsar=latsar)
            #subsetcoloc = subsetcoloc.drop('k') #to avoid issues of ambigiuity on k (whether coords or variable)
            subsetcoloc['crossSpectraImPol'] = crossSpectraImPol_xa
            subsetcoloc['crossSpectraImPol'].data = crossSpectraImPol_xa
            subsetcoloc['crossSpectraImPol'].attrs['description'] = 'variable from SLC WV products'

            subsetcoloc['crossSpectraRePol'] = crossSpectraRePol_xa
            subsetcoloc['crossSpectraRePol'].data = crossSpectraRePol_xa
            subsetcoloc['crossSpectraRePol'].attrs['description'] = 'variable from SLC WV products'
            subsetcoloc['py_S'] = xarray.DataArray(np.tile(S_slc.T, (1, 1)), dims=['time', 'N'],
                                                   coords={'time': times_bidons, 'N': np.arange(20)})
            subsetcoloc['py_S'].data = np.tile(S_slc.T, (1, 1)) # bug bizarre je suis oblige de faire ca pour ne pas avoir de NaN
            subsetcoloc['py_S'].attrs['description'] = 'S params from SLC polar xspectra'

            logging.info('crossSpectraRePol %s',crossSpectraRePol_xa.shape)
        #logging.info('ta : %s',ta)
        #logging.info('incidenceangle : %s',incidenceangle)
        lstvars_with_scale_factor_and_offset = ['hsALTmin','hsALTmax','incidenceAngle','hsALT','hsWW3','wsALTmin',
                                                'wsALT','wsALTmax','dx','dt','nk','nth','hsSM','h200','h400','h800',
                                                'trackAngle','hsWW3v2']
        for vvy in lstvars_with_scale_factor_and_offset :
            if vvy in subsetcoloc:
                subsetcoloc[vvy].encoding = {}
        #subsetcoloc = subsetcoloc.drop('k')  # to avoid ambiguous k coordinates definition
        for hh in subsetcoloc :
            if 'prb' in hh :
                subsetcoloc = subsetcoloc.drop(hh)



        #subsetcoloc['py_S'] = xarray.DataArray(S.T,dims=['time','N'],coords={'time':[datedt_slc],'N':np.arange(20)}) #solution simple
        #subsetcoloc['py_S'] = subsetcoloc['py_S'].attrs['description']='20 C-WAVE params computed from polar cross spectra 2-tau'
        all_subsets_coloc.append(subsetcoloc)
    return all_subsets_coloc

def save_training_file(dscoloc_enriched,outputfile):
    """

    :param dscoloc_enriched: contains py_S and X-spectra from SLC + Hs altimetric
    :param outputfile:
    :return:
    """
    # 5 ) save a netcdf file
    glob_attrs = {'processing_method': save_training_file.__name__,
                  'processing_script': os.path.basename(__file__),
                  'processing_env': sys.executable,
                  'processing_date': datetime.datetime.today().strftime('%Y%m%d %H:%M'),
                  'input_dir': 'IFREMER S1 WV SLC data + '+DIR_ORIGINAL_COLOCS,
                  'outputdir_dir': os.path.dirname(outputfile)
                  }
    dscoloc_enriched.attrs = glob_attrs
    dscoloc_enriched.attrs['created_on'] = '%s' % datetime.datetime.today()
    dscoloc_enriched.attrs['created_by'] = 'Antoine Grouazel'
    dscoloc_enriched.attrs['purpose'] = 'SAR Hs NN learning/inferences exp#1v4'
    dscoloc_enriched.attrs['purpose'] = 'content SAR & Alti colocations prepared by J.Stopa'
    logging.info('dscoloc_enriched : %s',dscoloc_enriched)
    for uu in dscoloc_enriched:
        logging.info('dtype %s %s',uu,dscoloc_enriched[uu].dtype)
    #dscoloc_enriched.to_netcdf(outputfile,encoding={'time':{'unit':'seconds since 2014-01-01 00:00:00'}})
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
    parser.add_argument('--slc',action='store',required=True,help='input slc (.tiff) file to treat ')
    parser.add_argument('--ocn',action='store',required=True,help='input ocn (.nc) file to treat ')
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
    datedt_slc = datetime.datetime.strptime(os.path.basename(args.slc).split('-')[4],'%Y%m%dt%H%M%S')
    outputfile_pattern = os.path.join(args.outputdir,datedt_slc.strftime('%Y'),
                              datedt_slc.strftime('%j'),'training_%s_*.nc' %os.path.basename(args.slc).replace('.tiff',''))
    logging.info('outputfile_pattern : %s',outputfile_pattern)
    if os.path.exists(os.path.dirname(outputfile_pattern)) is False :
        time.sleep(np.random.rand())
        os.makedirs(os.path.dirname(outputfile_pattern),0o0775)
    lst_output = glob.glob(outputfile_pattern)
    # remove is needed
    for outputfile in lst_output:
        if os.path.exists(outputfile) and args.redo:
            os.remove(outputfile)
    # skip if already present (on ne sais pas a l avance combien il a de matching)
    #if os.path.exists(outputfile) and args.redo is False:
    if len(lst_output)>=1 and args.redo is False: # je suppose que lun est fait il le son tous.
        logging.info('nothing to do, the file already exists')
        sys.exit(0)
    else:
        all_dscoloc_enriched = prepare_one_measurement(args.slc,args.ocn,dev=args.dev,add_cwave_L1=True,add_cwave_L2=True)
        for yyy,dscoloc_enriched in enumerate(all_dscoloc_enriched):
            outputfile = outputfile_pattern.replace('*',str(yyy+1).zfill(2))
            save_training_file(dscoloc_enriched,outputfile)
    logging.info('analysis done in %s seconds',time.time()-t1)
    logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)