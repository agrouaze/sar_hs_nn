# coding: utf-8
"""
copy paste from compute_hs_total_SAR_v2.py 31 January 2021

v2 is copy/paste from v1c, it is a CWAVE algo python that use the polSpec from OCN products,
it extracts the 20 param and use a CWAVE v2 model tuned on altimeters (only numpy dependencies) provided by Yannick Glaser (University of Hawaii)
date creation: 9 July 2019

C-WAVE params consist in 20 values computed from  Sentinel-1 C-band SAR WV image cross spectra

:env: export PYTHONPATH=~/sources/git/npCWAVE/
:purpose: methods to get total empirical hs from L2 SAR S-1 WV 
validated with
python 3.7.3
numpy                     1.13.1
scipy                     0.19.1
mkl                       2019.0

"""
import pdb
import logging
import numpy as np
import pandas as pd
import copy
import time
import os
import datetime
from sarhspredictor.lib.pol_cart_trans_jstopa_transcoding_furniture_cls import pol_cart_trans
import netCDF4
import traceback
reference_oswK_2015 = np.array([0.006283185, 0.007090763, 0.008002136, 0.00903065, 0.01019136, 
    0.01150126, 0.01297951, 0.01464776, 0.01653044, 0.01865509, 0.02105283, 
    0.02375875, 0.02681245, 0.03025866, 0.0341478, 0.03853681, 0.04348994, 
    0.0490797, 0.0553879, 0.06250691, 0.07054092, 0.07960752, 0.08983947, 
    0.1013865, 0.1144177, 0.1291238, 0.1457201, 0.1644495, 0.1855861, 
    0.2094395])
reference_oswK_2018 = np.array([0.005235988, 0.00557381, 0.005933429, 0.00631625, 0.00672377, 
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
reference_oswK_2017 = np.array([0.005235988, 0.00557381, 0.005933429, 0.00631625, 0.00672377, 
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

def prepare_input_CWAVE_vector_from_OCN(file_OCN):
    """

    :param file_OCN:
    :return:
    """
    osw_handler0 = netCDF4.Dataset(file_OCN)
    ths1,ks1,ta,incidenceangle,s0,nv,cspcRe,cspcIm,lonsar,latsar = read_input_from_OCN(osw_handler0)
    datedt = datetime.datetime.strptime(os.path.basename(file_OCN).split('-')[4],'%Y%m%dt%H%M%S')
    satellite = os.path.basename(file_OCN)[0:3]
    osw_handler0.close()
    return ths1,ks1,ta,incidenceangle,s0,nv,cspcRe,cspcIm,datedt,lonsar,latsar,satellite

def read_input_from_OCN(osw_handler0):
    """

    :param osw_handler0: (netCDF4.Dataset obj)
    :return:
    """
    ths1 = osw_handler0.variables['oswPhi'][:].squeeze()
    ks1 = osw_handler0.variables['oswK'][:].squeeze()
    lonsar = osw_handler0.variables['oswLon'][0][0]
    latsar = osw_handler0.variables['oswLat'][0][0]
    ta = osw_handler0.variables['oswHeading'][0][0]
    incidenceangle = osw_handler0.variables['oswIncidenceAngle'][0][0]
    s0 = osw_handler0.variables['oswNrcs'][0][0]
    nv = osw_handler0.variables['oswNv'][0][0]
    cspcRe = osw_handler0.variables['oswQualityCrossSpectraRe'][:].squeeze().T #transposition added by agrouaze
    cspcIm= osw_handler0.variables['oswQualityCrossSpectraIm'][:].squeeze().T #transposition added by agrouaze
    return ths1,ks1,ta,incidenceangle,s0,nv,cspcRe,cspcIm,lonsar,latsar

def format_input_CWAVE_vector_from_OCN(cspcRe,cspcIm,ths1,ta,incidenceangle,s0,nv,ks1,datedt,lonSAR,latSAR,satellite):
    """
    v2 is copy/paste from v1c, it is a CWAVE algo python that use the polSpec from OCN products, 
    it extracts the 20 param and use a CWAVE v2 model tuned on altimeters (only numpy dependancies)
    provided by Yannick Glaser (University of Hawaii)
    date creation: 9 Juillet 2019
    :args:
        dev_plots (bool):
        cspcRe (2D array): polSpecRe 60x72 (k,phi)
        cspcIm (2D array): polSpecIm 60x72 (k,phi)
        phisar (1D array): oswPhi
        s0 (float): oswNrcs
        ks1 (float): oswK
        internalvars_v1 (str): ful path pkl file [optional]
        satellite (str): s1a or...
    :returns:
        hs_total_sar (float): in meters
        
        
    % example 1:
    %   s1a-wv1-ocn-vv-20151130t175854-20151130t175857-008837-00c9eb-087.nc
    % hsSM=3.4485
    
    % example 2:
    %   s1a-wv2-ocn-vv-20151130t201457-20151130t201500-008838-00c9f5-046.nc
    % hsSM=1.3282
    """
    t0 = time.time()
    logging.debug('start computing hs total SAR with X spectra shape %s',cspcRe.shape)
#     % Constants for CWAVE======================================================
    NTH = 72    # % number of output wave directions on log-polar grid
    kmax=2*np.pi/60 #  % kmax for empirical Hs
    kmin=2*np.pi/625 # % kmin for empirical Hs
    ns = 20     # % number of variables in orthogonal decomposition
    S = np.ones((ns,1))*np.nan
    dky = 0.002954987953815
    nky = 85
    dkx = 0.003513732113299
    nkx = 71
   
    kx = (np.arange(0,nkx).T-np.floor(nkx/2))*dkx#I remove the minus one in the length agrouaze
    logging.debug('kx =  %s',kx.shape)
    logging.debug('np.floor(nkx/2)+1 = %s',np.floor(nkx/2)+1)
    kx[int(np.floor(nkx/2))] = 0.0001 #agrouaze remove the +1 to be like in matlab
    ky = (np.arange(0,nky).T-np.floor(nky/2))*dky #I remove the minus one in the length agrouaze
    ky[int(np.floor(nky/2))] = 0.0001 #agrouaze remove the +1 to be like in matlab
        
    logging.debug('ky = %s',ky.shape)
#     KY,KX = scipy.ndgrid(kx,ky)
    #if False: #output is 85,71 instead of 71,85
    #    KY,KX = np.meshgrid(kx,ky)
    #else:
    KX,KY = np.meshgrid(ky,kx)
    logging.debug('KY shape = %s KX shape = %s',KY.shape,KX.shape)
    condition_keep = (abs(KX)>=kmin) & (abs(KX)<=kmax) & (abs(KY)>=kmin) & (abs(KY)<=kmax)
#     condition_keep = (abs(KX)>=kmin) | (abs(KX)<=kmax) | (abs(KY)>=kmin) | (abs(KY)<=kmax)
    indices = np.where(condition_keep)
    logging.debug('indices %s',indices)
    gdx,gdy = indices
    logging.debug('gdx = %s gdy = %s',gdx.shape,gdy.shape)
    logging.debug('gdx[0] = %s %s',gdx[0:5],gdy[0:5])
    #I skip the unique since it leads to an error because the to vectors gdx and gdy have not the same size after unicity
#     gdx=np.unique(gdx)
#     gdy=np.unique(gdy)
    assert KX.shape==(71,85)
#     logging.debug('after unique gdx = %s gdy = %s',gdx.shape,gdy.shape)
    v = 1
    if v==1:
        KX = KX[gdx,gdy] #agrouaze test to fit to matlab inverse gdx and gdy
        KY = KY[gdx,gdy]
        uniq_gdx = np.unique(gdx)#54x1 in matlab
        uniq_gdy = np.unique(gdy)#64x1 in matlab
        KX = np.reshape(KX,(len(uniq_gdx),len(uniq_gdy)))
        KY = np.reshape(KY,(len(uniq_gdx),len(uniq_gdy)))
#         logging
    elif v==2:#https://stackoverflow.com/questions/5819118/slicing-in-python-similar-to-matlab
        KX = KX[np.r_[gdx],np.r_[gdy]]
        KY = KY[np.r_[gdx],np.r_[gdy]]
    elif v==3:
        #idea mathias fillvalues
        KX = np.ma.masked_where(condition_keep==False, KX, copy=True)
        KY = np.ma.masked_where(condition_keep==False, KY, copy=True)
        
    
    logging.debug('after subsampling KY shape = %s KX shape = %s',KY.shape,KX.shape)
#     KX = KX[indices]
#     KY = KY[indices]
    DKX = np.ones(KX.shape)*0.003513732113299
    DKY = np.ones(KX.shape)*0.002954987953815

    if (ks1>1000).any():
        flagKcorrupted = True
    else:
        flagKcorrupted = False
    logging.debug('flagKcorrupted = %s',flagKcorrupted)
    if (ks1>1000).any()and False:#improv agrouaze turned off while the validaiton is not over
        logging.info('beware oswK contains fillvalues')
        indices_pourris = (ks1>1000)
        ks1[indices_pourris] = reference_oswK_2017[indices_pourris]
    ia = incidenceangle
    logging.debug('cspcRe = %s',cspcRe.shape)
    subset_ok = {}
    subset_ok['todSAR'] = _conv_time(netCDF4.date2num(datedt,'hours since 2010-01-01T00:00:00Z UTC'))
    subset_ok['lonSAR'] = lonSAR
    subset_ok['latSAR'] = latSAR
    subset_ok['incidenceAngle'] = incidenceangle
    subset_ok['sigma0'] = s0
    subset_ok['normalizedVariance'] = nv
    if cspcRe.shape[0] == 60 and (cspcRe>0).any():
        # Convert to kx,ky spectrum
        a1 = np.radians(ta)
        a2 = np.radians(ths1)
        dif = np.arctan2(np.sin(a2-a1),np.cos(a2-a1))
        logging.debug('dif = %s',dif)
#         [~,str]=min(abs(dif))
        strr = np.argmin(abs(dif))
        idd = np.mod(np.arange(strr,strr+NTH),NTH) #minus -1 added to the modulo (agrouaze) since in python indices are starting at 0 and finishing at len-1
        logging.debug('idd = %s %s',idd,len(idd))
        logging.debug('cross spectra before subsetting %s',cspcRe.shape)
        logging.debug('cross spectra after subsetting %s',cspcRe[:,idd].shape)

        interpmethod = 'linear' #D max diff 38 too smooth 0.05m diff (difference au centre du plot (k tres petit)
#         interpmethod = 'nearest' #D max diff 25 too smooth but worst hs 0.14m diff
#         interpmethod = 'cubic' #D max diff 28 too smooth but worst hs 0.05m diff

        cspcReX,cspcReX_not_conservativ = pol_cart_trans(d=cspcRe[:,idd],k=ks1,t=np.radians(ths1),x=kx,y=ky,
                                                             name='re',interpmethod=interpmethod)
        assert cspcReX.shape==(71,85) #or (cspcReX.shape==(85,71))) #info de justin
        assert cspcReX.size==71*85
        cspcImX,cspcImX_not_conservativ = pol_cart_trans(d=cspcIm[:,idd],k=ks1,t=np.radians(ths1),x=kx,y=ky,
                                                             name='im',interpmethod=interpmethod)
            
        logging.debug('cspcReX = %s',cspcReX.shape)
        #test pour voir si le dernire pb est bien l interpolation pol->cart: reponse oui
#         logging.warning('matlab reaplcement cross spectra')
#         cspcReX = mat['cspcReX']
#         cspcImX = mat['cspcImX']
        cspc = np.sqrt(cspcReX**2+cspcImX**2)
        logging.debug('cspc %s',cspc.shape)
        logging.debug('gdx = %s max = %s',gdx.shape,gdx.max())
        logging.debug('gdy = %s max = %s',gdy.shape,gdy.max())
#         cspc = cspc[gdx,gdy,:]
        if v==2: #error too many indices for array
            cspc = cspc[gdx,gdy,:]
        elif v==3:#with fillvalues
            cspc = np.ma.masked_where(condition_keep==False,cspc)
        elif v==1:#with reshape after np.where
            cspc = cspc[gdx,gdy] #agrouaze
            #cspc = cspc[gdy,gdx]
            cspc = np.reshape(cspc,(len(uniq_gdx),len(uniq_gdy))) #agrouaze added
        logging.debug('after subsampling cspc %s',cspc.shape)
            
        #       % Compute Orthogonal Moments===============================================
        gamma = 2
        a1 = (gamma**2-np.power(gamma,4))/(gamma**2*kmin**2-kmax**2)
        a2 = (kmax**2-np.power(gamma,4)*kmin**2)/(kmax**2-gamma**2*kmin**2)
        
#         % Ellipse
        tmp = a1*np.power(KX,4)+a2*KX**2+KY**2
#         % eta
        eta = np.sqrt( (2.*tmp)/((KX**2+KY**2)*tmp*np.log10(kmax/kmin)) )
        logging.debug('eta = %s',eta.shape)

        alphak = 2.*(( np.log10(np.sqrt(tmp))-np.log10(kmin) )/np.log10(kmax/kmin) )-1
#         alphak = alphak.transpose() #added by agrouaze to be in the same config as matlab
#        % Fix alphak
        alphak[(alphak**2)>1] = 1.
            
        #if True: #different matrix compare to matlab
        alphat = np.arctan2(KY,KX)
        #else:
        #    alphat = np.arctan2(KX,KY)
        
        logging.debug('alphat = %s',alphat.shape)

        
        # Gegenbauer polynomials
        tmp = abs(np.sqrt(1-alphak**2))# % imaginary???

        g1 = 1/2.*np.sqrt( 3)*tmp
        g2 = 1/2.*np.sqrt(15)*alphak*tmp
#         g3 = (1/4.)*np.sqrt(7/6)*(15.*np.power(alphak,2)-3.)*tmp
        g3pre = np.double(1/4.)*np.sqrt(7./6.)*(15.*np.power(alphak,2)-np.double(3.))
        g3 = np.dot((1/4.)*np.sqrt(7./6.),(15.*np.power(alphak,2)-3.))*tmp #
        #g3 = 1/(4.*np.sqrt(7/6)*(15.*np.power(alphak,2)-3.)*tmp) #faux 0.1 diff
#         g3 = np.dot(((1/4.)*np.sqrt(7/6)*(15.*np.power(alphak,2)-3)).T,tmp)
        g4 = (1/4.)*np.sqrt(9./10)*(35.*np.power(alphak,3)-15.*alphak**2)*tmp
        gi = {'g1':g1,'g2':g2,'g3':g3,'g4':g4}
        logging.debug('g1 = %s',g1.shape)
        
        # Harmonic functions
        f1=np.sqrt(1/np.pi)*np.cos(0.*alphat)
        f2=np.sqrt(2/np.pi)*np.sin(2.*alphat)
        f3=np.sqrt(2/np.pi)*np.cos(2.*alphat)
        f4=np.sqrt(2/np.pi)*np.sin(4.*alphat)
        f5=np.sqrt(2/np.pi)*np.cos(4.*alphat)
#         logging.debug('f1 = %s,f2 = %s,f3 = %s,f4 = %s f5= %s',f1,f2,f3,f4,f5)
        
        #       % Weighting functions
        logging.debug('KX shape = %s',KX.shape)
        h = np.ones((KX.shape[0],KX.shape[1], 20))
        logging.debug('h computation g1 = %s f1 = %s eta= %s',g1.shape,f1.shape,eta.shape)
        h[:,:,0]=g1*f1*eta #agrouaze transpose gi
#         h[:,:,0]= np.dot(np.dot(g1,f1),eta)
        h[:,:,1] = g1*f2*eta
        h[:,:,2] = g1*f3*eta
        h[:,:,3] = g1*f4*eta
        h[:,:,4] = g1*f5*eta
        h[:,:,5] = g2*f1*eta
        h[:,:,6] = g2*f2*eta
        h[:,:,7] = g2*f3*eta
        h[:,:,8] = g2*f4*eta
        h[:,:,9] = g2*f5*eta
        h[:,:,10] = g3*f1*eta
        h[:,:,11] = g3*f2*eta
        h[:,:,12] = g3*f3*eta
        h[:,:,13] = g3*f4*eta
        h[:,:,14] = g3*f5*eta
        h[:,:,15] = g4*f1*eta
        h[:,:,16] = g4*f2*eta
        h[:,:,17] = g4*f3*eta
        h[:,:,18] = g4*f4*eta
        h[:,:,19] = g4*f5*eta

        logging.debug('S shape = %s',S.shape)
        try:
          
            P = cspc/(np.nansum(np.nansum(cspc*DKX*DKY))) #original
        except:
            logging.error('%s',traceback.format_exc())
            logging.error('date = %s',datedt)
            logging.error('cspc = %s %s %s all dv = %s mask all True? %s',cspc.shape,cspc.dtype,type(cspc),(np.isfinite(cspc)==False).all(),(cspc.mask==True).all())
            cspc_tmp2 = copy.copy(cspc)
            cspc_tmp2.mask=False
            logging.error('without mask cspc = %s %s',cspc_tmp2.min(),cspc_tmp2.max())
            logging.error('DKX = %s %s',DKX,DKX.shape)
            logging.error('DKY = %s %s',DKY,DKY.shape)
            logging.error('tst %s',cspc*DKX*DKY)
            logging.error('tst2 %s',np.nansum(cspc*DKX*DKY).shape)
            cspc_masked = np.ma.masked_where(np.isnan(cspc),cspc,copy=True)
            denom = np.sum(cspc_masked*DKX*DKY)
            logging.error('denom = %s',denom)
            P = cspc_masked/denom
            P = np.array([1]) #trick to return something but not usable
            flagKcorrupted = True
        if np.isfinite(s0) is False or isinstance(s0,np.ma.core.MaskedConstant):# or s0.mask==True:
            s0 = 0
            flagKcorrupted = True
        if np.isfinite(nv)is False or isinstance(nv,np.ma.core.MaskedConstant):# or nv.mask==True:
            nv = 0
            flagKcorrupted = True
        if np.isfinite(incidenceangle) is False or isinstance(incidenceangle,np.ma.core.MaskedConstant):# or incidenceangle.mask==True:
            incidenceangle = 0
            flagKcorrupted = True
            
            
        logging.debug('ns = %s',ns)
        logging.debug('P shape = %s',P.shape)
        logging.debug('h shape = %s',h.shape)
        for jj in range(ns):
            petit_h = h[:,:,jj].squeeze().T #added by agrouaze to be like in matlab code
            logging.debug('S computation = petit_h=%s P = %s DKX = %s DKY = %s',petit_h.shape,P.shape,DKX.shape,DKY.shape)
            S[jj]=np.nansum(np.nansum(petit_h*P.T*DKX.T*DKY.T))
        
        logging.debug('S = %s',len(S))
        
        logging.debug('S = %s',len(S))
        
        var_to_keep = ['todSAR', 'lonSAR', 'latSAR', 'incidenceAngle', 'sigma0',
      'normalizedVariance', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
       's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
       's18', 's19', 'sentinelType']
        

        for iiu in range(len(S)):
            subset_ok['s'+str(iiu)] = S[iiu][0]
        # encodes type A as 1 and B as 0
        if satellite=='s1a':
            subset_ok['sentinelType'] = 1
        else:
            subset_ok['sentinelType'] = 0
        logging.debug('subset_ok = %s',subset_ok)
        try:
            subset_ok = pd.DataFrame(subset_ok,index=[0])
        except:
            logging.error('impossible to convert dict subset_ok into dataframe pandas %s',traceback.format_exc())
            logging.info('let the subset_ok as dict')
    else:
        cspcReX = np.zeros((71,85))
        cspcImX = np.zeros((71,85))
        cspcReX_not_conservativ = np.zeros((71,85))
        #cspcReX = np.array([])
        #cspcImX = np.array([])
        #cspcRe = np.array([])
        #cspcReX_not_conservativ = np.array([])
    logging.debug('subset_ok %s',subset_ok)
    return subset_ok,flagKcorrupted,cspcReX,cspcImX,cspcRe,ks1,ths1,kx,ky,cspcReX_not_conservativ,S#[:,idd]

def _conv_time(in_t):
    """
    Converts data acquisition time

    Args:
        in_t: time of data acquisition in format hours since 2010-01-01T00:00:00Z UTC

    Returns:
        Encoding of time where 00:00 and 24:00 are -1 and 12:00 is 1
    """
    in_t = in_t % 24
    return 2 * np.sin((2 * np.pi * in_t) / 48) - 1



