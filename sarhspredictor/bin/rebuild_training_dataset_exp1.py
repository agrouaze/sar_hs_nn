"""
exp 1: doing the Quach experiment with X spectra from SLC instead of OCN
 steps 1) read a tiff 2) find the indice of this tiff in the cwaveV4 to retrieve Hs alti 3) compute 22 CWAVE params from SLC cross spectra 4) add SLC cross spectra
 (note that I could keep the CWAVE params from L2 xspectra... not tested yet)
May 2021
A Grouazel
the listing of tiff to treat is here: /home1/datawork/agrouaze/data/sentinel1/cwave/listing_SAR_L2_L1_measu_from_colocations_cwaveV4.txt
inspiration rebuild_training_dataset.py
tested with cwave

"""
import os
import sys
sys.path.append('/home1/datahome/agrouaze/git/xsarseafork/src/xsarsea')
sys.path.append('/home1/datahome/agrouaze/git/sar_hs_nn')
import logging
import xarray
import numpy as np
import datetime
import xsarsea
import xsarsea.cross_spectra_core
import warnings
import copy
import time
from collections import defaultdict
import resource
warnings.simplefilter(action='ignore',category=FutureWarning)
from scipy.spatial import KDTree
import traceback

import pdb

from predict_hs_from_SLC import compute_Cwave_params_and_xspectra_fromSLC

DIR_ORIGINAL_COLOCS = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/quach2020/validation/colocations/original_colocations_YoungAltiDatabase_vs_WV_L2_Jstopa/cwaveV4/'
OUTPUTDIR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp1/training_dataset/v1/'
def round_seconds(obj: datetime.datetime) -> datetime.datetime:
    if obj.microsecond >= 500_000:
        obj += datetime.timedelta(seconds=1)
    return obj.replace(microsecond=0)


def find_indices_sar_acquisition_in_monthly_colocs(ds_coloc,datedt_ocn,lon_x,lat_x,new_times):
    """ds_coloc
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
            inds = tmp_inds[0][0]
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

    :param slc:
    :param ocn:
    :return:
    """
    datedt_ocn = datetime.datetime.strptime(os.path.basename(slc).split('-')[4],'%Y%m%dt%H%M%S')
    dsocn = xarray.open_dataset(ocn)
    lon_x = dsocn['oswLon'].values.squeeze()
    lat_x = dsocn['oswLat'].values.squeeze()
    sat = os.path.basename(slc)[0 :3]
    coloc_file_JS = os.path.join(DIR_ORIGINAL_COLOCS,
                                 sat.upper() + '_ALT_coloc' + datedt_ocn.strftime('%Y%m') + 'S.nc')
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
    return new_times,lon_x,lat_x,sat,ds_coloc,datedt_ocn



def prepare_one_measurement(slc,ocn,dev=False):
    """

    :param slc: str
    :param ocn : str
    :param coloc_file_JS: str full path of the colocation file from Justin Stopa
    :return:
    """
    logging.debug('slc :%s',slc)
    logging.debug('ocn: %s',ocn)

    new_times,lon_x,lat_x,sat,ds_coloc,datedt_ocn = read_coloc_file(slc,ocn)
    inds,ds_coloc,cptx = find_indices_sar_acquisition_in_monthly_colocs(ds_coloc,datedt_ocn,lon_x,lat_x,new_times)
    nb_match = len(inds)
    # 1) read all params from Justin s dataset
    subsetcoloc = ds_coloc.isel(time=inds) #{ on peut selectionner plusieurs indice en meme temps avec isel}

    logging.info('subsetcoloc : %s',subsetcoloc)
    # keep a time dimension to be sure to be able to gather with open_mfdataset
    for vv in subsetcoloc.keys():
        logging.debug('vv: %s %s',vv,subsetcoloc[vv].shape)
        if vv in ['S','cspcRe','cspcIm']:
            subsetcoloc = subsetcoloc.drop(vv)
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

    # 4) compute C wave params
    ths1 = np.arange(0,360,5)
    ta = subsetcoloc['trackAngle'].values[0]
    s0 = subsetcoloc['sigma0'].values[0]
    nv = subsetcoloc['normalizedVariance'].values[0]
    incidenceangle = subsetcoloc['incidenceAngle'].values[0]
    lonsar = subsetcoloc['lonSAR'].values[0]
    latsar = subsetcoloc['latSAR'].values[0]

    crossSpectraImPol_xa,crossSpectraRePol_xa,times_bidons,S = compute_Cwave_params_and_xspectra_fromSLC(slc,dev,
                    nb_match=1,ths1=ths1,ta=ta,s0=s0,nv=nv,incidenceangle=incidenceangle,lonsar=lonsar,latsar=latsar)
    subsetcoloc = subsetcoloc.drop('k') #to avoid issues of ambigiuity on k (whether coords or variable)
    subsetcoloc['crossSpectraImPol'] = crossSpectraImPol_xa
    subsetcoloc['crossSpectraRePol'] = crossSpectraRePol_xa
    logging.info('crossSpectraRePol %s',crossSpectraRePol_xa.shape)
    logging.info('ta : %s',ta)
    logging.info('incidenceangle : %s',incidenceangle)
    lstvars_with_scale_factor_and_offset = ['hsALTmin','hsALTmax','incidenceAngle','hsALT','hsWW3','wsALTmin',
                                            'wsALT','wsALTmax','dx','dt','nk','nth','hsSM','h200','h400','h800',
                                            'trackAngle','hsWW3v2']
    for vvy in lstvars_with_scale_factor_and_offset :
        if vvy in subsetcoloc:
            subsetcoloc[vvy].encoding = {}
    subsetcoloc = subsetcoloc.drop('k')  # to avoid ambiguous k coordinates definition
    for hh in subsetcoloc :
        if 'prb' in hh :
            subsetcoloc = subsetcoloc.drop(hh)

    subsetcoloc['py_S'] = xarray.DataArray(np.tile(S.T,(nb_match,1)),dims=['time','N'],coords={'time' : times_bidons,'N' : np.arange(20)})
    #subsetcoloc['py_S'] = xarray.DataArray(S.T,dims=['time','N'],coords={'time':[datedt_slc],'N':np.arange(20)}) #solution simple
    #subsetcoloc['py_S'] = subsetcoloc['py_S'].attrs['description']='20 C-WAVE params computed from polar cross spectra 2-tau'
    return subsetcoloc

def save_training_file(dscoloc_enriched,outputfile):
    """

    :param dscoloc_enriched: contains py_S and X-spectra from SLC + Hs altimetric
    :param outputfile:
    :return:
    """
    # 5 ) save a netcdf file
    dscoloc_enriched.attrs['created_on'] = '%s' % datetime.datetime.today()
    dscoloc_enriched.attrs['created_by'] = 'Antoine Grouazel'
    dscoloc_enriched.attrs['purpose'] = 'SAR Hs NN learning/inferences exp#1'
    dscoloc_enriched.attrs['purpose'] = 'content SAR & Alti colocations prepared by J.Stopa'
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
    datedt_slc = datetime.datetime.strptime(os.path.basename(args.slc).split('-')[4],'%Y%m%dt%H%M%S')
    outputfile = os.path.join(outdir,datedt_slc.strftime('%Y'),
                              datedt_slc.strftime('%j'),'training_%s.nc' %os.path.basename(args.slc).replace('.tiff',''))
    logging.info('outputfile : %s',outputfile)
    if os.path.exists(os.path.dirname(outputfile)) is False :
        os.makedirs(os.path.dirname(outputfile),0o0775)
    if os.path.exists(outputfile) and args.redo:
        os.remove(outputfile)
    if os.path.exists(outputfile) and args.redo is False:
        logging.info('nothing to do, the file already exists')
        sys.exit(0)
    else:
        dscoloc_enriched = prepare_one_measurement(args.slc,args.ocn,dev=args.dev)
        save_training_file(dscoloc_enriched,outputfile)
    logging.info('analysis done in %s seconds',time.time()-t1)
    logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)