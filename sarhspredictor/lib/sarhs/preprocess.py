# Define utility functions for preprocessing SAR data.
import numpy as np
import datetime
#PERCENTILE_99 = {'re':4715029695.5043869,'im':595082012.33398867} # performed on v3 exp1
#PERCENTILE_99 = {'re':1.5377643610847046e+17,'im':82147297185.429794} # estimation performed on v4 exp1 Feb 2022, not used since the values seems too high compare to previous v3, risk to degrade the spectrum
PERCENTILE_99 = {'re':4.124100750609777e+28,'im':6.357411979053802e+22} #estimated le 12 may 2022 with exp1v4/v6, apres generation des fichiers normalises je vois que la normalization est un peu forte.. mais il reste du signal.. a voir si on parviens à faire du training avec
COEFS_clipping_and_linear_scaling = { # estimated 18 may 2022 on exp1v4/v6 with a dataset usin xsarsea version without any normalization integrated estimation_SLC_spectrum_normalization_and_outlier.ipynb
    're_clip': 5.435146994785513e+19, # 7*99e percentile 1.3% data clipped
    're_offset' : 34048801642485.62,
    're_fact' : 3.651009800615139e+17, # 70e percentile -> std 8.2 et maximum 149
    'im_clip':4.670750837386242e+18, #7*99e percentile 1.5% data clipped
    'im_offset' : 7734889022997.355,
    'im_fact':3.0584672481884656e+16, #70e percentile -> std 6.1 et maximum 153
 }

COEFS_clipping_and_linear_scaling_v2 = { # estimated 24 may 2022 on exp1v4/v7 with a dataset usin xsarsea version without any normalization integrated estimation_SLC_spectrum_normalization_and_outlier.ipynb
    're_clip': 5.8124298085085454e+19, # 7*99e percentile 1.6% data clipped
    're_offset' : 0, # due to clipping negative values to zeros
    're_fact' : 3.60540621201969e+17, # 70e percentile -> std 9.2 et maximum 161
    'im_clip':9.66795441691413e+18, # 25*99e percentile
    'im_offset' : None,
    'im_fact':1e17 #-> std 2.1 maxi +/-77'
 }
# better to loose few very high spectrum than lots of spectrum close to the mean
def _conv_timeofday(in_t):
    """Converts data acquisition time
    Args:
        in_t: time of data acquisition in format hours since 2010-01-01T00:00:00Z UTC
    Returns:
        Encoding of time where 00:00 and 24:00 are -1 and 12:00 is 1
    """
    in_t = in_t%24
    return 2*np.sin((2*np.pi*in_t)/48)-1

def _conv_deg(in_angle, is_inverse=False, in_cos=None, in_sin=None):
    """Converts measurements in degrees (e.g. angles), using encoding proposed at https://stats.stackexchange.com/a/218547
       Encode each angle as tuple theta as tuple (cos(theta), sin(theta)), for justification, see graph at bottom
    Args:
        coord: measurement of lat/ long in degrees
    Returns:
        tuple of values between -1 and 1
    """
    if is_inverse:
        return np.sign(np.rad2deg(np.arcsin(in_sin))) * np.rad2deg(np.arccos(in_cos))
    
    angle = np.deg2rad(in_angle)
    return (np.cos(angle), np.sin(angle))

def apply_clipping_and_linear_scaling(re_spec,clipping_val,factor):
    """

    :param re_spec: spectra 72x60
    :param clipping_val: flaot
    :param factor: float
    :return:
    """
    #clipping instead of filtering
    re_spec4 = re_spec.copy()
    re_spec4[re_spec>clipping_val] = clipping_val
    #np_spectra_with_higher_val = np.any(re_spec>clipping_val,axis=1).sum()
    #pct = 100*np_spectra_with_higher_val/re_spec.shape[0]
    val_2 = np.nanmin( re_spec4)
    re_spec5 = re_spec4-val_2
    #val_3 = np.nanpercentile(re_spec5.ravel(),percentile)
    re_spec6 = re_spec5/factor
    return clipping_val,val_2,factor,re_spec6

def apply_clipping_and_linear_scaling_im(im_spec3,percentile=None,divide_fact=None,clipping_val=None):
    pct_cliped = np.nan
    im_spec4 = im_spec3.copy()
    if clipping_val is not None and np.isfinite(clipping_val):
        im_spec4[abs(im_spec3)>clipping_val] = clipping_val
        np_spectra_with_higher_val = np.any(abs(im_spec3)>clipping_val,axis=1).sum()
        pct_cliped = 100*np_spectra_with_higher_val/im_spec3.shape[0]
        print('nb spec with clipped values',np_spectra_with_higher_val,pct_cliped)
    #val_2 = np.nanmin( im_spec4)
    val_2 = 0 #je test sans le decalage qui semble inutile
    im_spec5 = im_spec4-val_2
    if percentile is not None:
        val_3 = np.nanpercentile(im_spec5.ravel(),percentile)
    else:
        val_3 = divide_fact
    im_spec6 = im_spec5/val_3
    return clipping_val,percentile,val_2,val_3,im_spec6,pct_cliped

def apply_clipping_and_linear_scaling_re(re_spec3,percentile=None,divide_fact=None,clipping_val=None):
    """

    :param re_spec3: nd.array
    :param percentile: int to find the dividing factor
    :param divide_fact: float to give directly a value insted of a percentile
    :param clipping_val: float
    :return:
    """
    # test normalization
    #on clip a zero en bas
    pct_clipped = np.nan
    re_spec3[re_spec3<0] = 0. #the extrapolation  int conversion_polar_cartesian() with kwargs={"fill_value": None} introduce values below zero for 0.27% of the pts
     #clipping instead of filtering
    re_spec4 = re_spec3.copy()
    if clipping_val is not None and np.isfinite(clipping_val):
        re_spec4[re_spec3>clipping_val] = clipping_val
        np_spectra_with_higher_val = np.any(re_spec3>clipping_val,axis=1).sum()
        pct_clipped = 100*np_spectra_with_higher_val/re_spec3.shape[0]
    val_2 = np.nanmin( re_spec4)
    re_spec5 = re_spec4-val_2
    if percentile is not None:
        val_3 = np.nanpercentile(re_spec5.ravel(),percentile)
    else:
        val_3 = divide_fact
    re_spec6 = re_spec5/val_3
    return clipping_val,percentile,val_2,val_3,re_spec6,pct_clipped



def conv_real(x,exp_id=None):
    """Scales real part of spectrum.
    Args:
        real: numpy array of shape (notebooks, 72, 60)
    Returns:
        scaled
    """
    assert len(x.shape) == 3
    assert x.shape[1:] == (72, 60)
    if exp_id is None:
        x = (x - 8.930369) / 41.090652
    elif exp_id==1:
        #x = x / 1150179.354676391 # le min etant en e-30 je me permet de ne pas le remettre a zero (tested in stats_dataset_training_exp1.ipynb)
        #x = x / 266297713.37484202
        #x = x /268312468.14585936
        #x = x / 2106218840.5231066 #update 27 sept a partir des spectre SLC dans la v3 des fichiers individuels pour generer D1 (stats_dataset_training_exp1.ipynb)
        #x = x / 9618124186.148272 # update 1st oct 2021 apres investigation complete de 2018 dataset brut
        #x = x / 2.83349121369e+14 # update 4th oct 2021 apres investigattion complete de tout le dataset
        #x = x / PERCENTILE_99['re'] # 99 percentile
        # _,_,_,x = apply_clipping_and_linear_scaling(x, clipping_val=COEFS_clipping_and_linear_scaling['re_clip'],
        #                                               factor=COEFS_clipping_and_linear_scaling['re_fact'])
        _,_,_,_,x,_ = apply_clipping_and_linear_scaling_re(x, clipping_val=COEFS_clipping_and_linear_scaling_v2['re_clip'],
                                                      divide_fact=COEFS_clipping_and_linear_scaling_v2['re_fact'])
    else:
        raise Exception('unkown exp_id %s'%exp_id)

    return x

def conv_imaginary(x,exp_id=None):
    """Scales imaginary part of spectrum.
    Args:
        real: numpy array of shape (notebooks, 72, 60)
    Returns:
        scaled
    """
    assert len(x.shape) == 3
    assert x.shape[1:] == (72, 60)
    if exp_id is None :
        x = (x - 4.878463e-08) / 6.4714637
    elif exp_id == 1: #evaluate in stats_dataset_training_exp1.ipynb
        #x = x / 330446.8692954672  # le min etant en e-30 je me permet de ne pas le remettre a zero
        #x = x/6605101.291873287
        #x = x/156310666.1937346 #update 27 sept a partir des spectre SLC dans la v3 des fichiers individuels pour generer D1 (stats_dataset_training_exp1.ipynb) (2287 spectres sample)
        #x = x / 863468236.6305181 # update 1st oct 2021 apres investigation complete du dataset brut
        #x = x / 1.52703468968e+11 # update 4th oct 2021 apres investigattion complete de tout le dataset
        #x = x / PERCENTILE_99['im'] # 99 percentile
        # _,_,_,x = apply_clipping_and_linear_scaling(x, clipping_val=COEFS_clipping_and_linear_scaling['im_clip'],
        #                                               factor=COEFS_clipping_and_linear_scaling['im_fact'])
        _,_,_,_,x,_ = apply_clipping_and_linear_scaling_im(x, clipping_val=COEFS_clipping_and_linear_scaling_v2['im_clip'],
                                                      divide_fact=COEFS_clipping_and_linear_scaling_v2['im_fact'])
    else :
        raise Exception('unkown exp_id %s' % exp_id)

    return x
 

def median_fill(x, extremum=1e+15):
    """
    Inplace median fill.
    Args:
    x: numpy array of shape (notebooks, features)
    extremum: threshold for abs value of x. Damn Netcdf fills in nan values with 9.96921e+36.
    Returns:
    rval: new array with extreme values filled with median.
    """
    #assert not np.any(np.isnan(x)) #commented by agrouaze Feb 2021
    medians = np.median(x, axis=0)
    mask = np.abs(x) > extremum
    medians = np.repeat(medians.reshape(1,-1), x.shape[0], axis=0)
    assert medians.shape == x.shape, medians.shape
    x[mask] = medians[mask] # TODO: MODIFIES x, so this is unsafe.
    return x

def conv_cwave(x):
    """
    Scale 22 cwave features. These were precomputed using following script.
    
    from sklearn import preprocessing
    with h5py.File('aggregate_ALT.h5', 'r') as fs:
        cwave = np.hstack([fs['S'][:], fs['sigma0'][:].reshape(-1,1), fs['normalizedVariance'][:].reshape(-1,1)])
        cwave = scripts.median_fill(cwave) # Needed to remove netcdf nan-filling.
        s_scaler = preprocessing.StandardScaler()
        s_scaler.fit(cwave) # Need to fit to full data.
        print(s_scaler.mean_, s_scaler.v)
    
    """
    # Fill in extreme values with medians.
    x = median_fill(x)
    
    means = np.array([ 8.83988852e+00,  9.81496891e-01,  2.04964720e+00,  1.05590932e-01,
        -6.00710228e+00,  2.54775182e+00, -5.76860655e-01,  2.09000078e+00,
        -8.44825896e-02,  8.90420253e-01, -1.44932907e+00, -6.79597846e-01,
         1.03999407e+00, -2.09475628e-01,  2.76214306e+00, -6.35718150e-03,
        -8.09685487e-01,  1.41905445e+00, -1.85369068e-01,  3.00262098e+00,
        -1.06865787e+01,  1.33246124e+00])
    
    vars = np.array([ 9.95290027, 35.2916408 ,  8.509233  , 10.62053105, 10.72524569,
         5.17027335,  7.04256618,  3.03664677,  3.72031389,  5.92399639,
         5.31929415,  8.26357553,  1.95032647,  3.13670466,  3.06597742,
         8.8505963 , 13.82242244,  1.43053089,  1.96215081, 11.71571483,
         27.14579017,  0.05681891])
    
    x = (x - means) / np.sqrt(vars)
    return x

def conv_dx(dx):
    """
    Scale dx (distance between SAR and ALT) by std. Computed with:
    
    with h5py.File('aggregate_ALT.h5', 'r') as fs:
        dd = np.hstack([fs['dx'][:].reshape(-1,1), fs['dt'][:].reshape(-1,1)])
        print(dd.std(axis=0))
    """
    return dx / 55.24285431 

def conv_dt(dt):
    """
    Scale dt (time diff between SAR and ALT) by std. Computed with:
    
    with h5py.File('aggregate_ALT.h5', 'r') as fs:
        dd = np.hstack([fs['dx'][:].reshape(-1,1), fs['dt'][:].reshape(-1,1)])
        print(dd.std(axis=0))
    """
    return dt / 36.70367443

def conv_position(latSAR):
    """
    Return cosine and sine to latitute/longitude feature.
    """
    coord_transf = np.vectorize(_conv_deg)
    cos, sin = coord_transf(latSAR)
    return np.column_stack([cos, sin])
    
def conv_time(timeSAR):
    """
    Return time of day feature.
    """
    time_transf = np.vectorize(_conv_timeofday)
    time_of_day = time_transf(timeSAR)
    #return np.column_stack(timeSAR, time_of_day)
    return time_of_day

def conv_time_doy(timeSAR):
    """
    Return day or year feature.
    :param :timeSAR datetime obj
    """
    doys = []
    for tti in timeSAR:
        ts = (tti - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        ts2 = datetime.datetime.utcfromtimestamp(ts)
        doy = int(ts2.strftime('%j'))
        doys.append(doy)
    timeSAR = np.array(doys)
    day_of_year = timeSAR / 365.  #
    return day_of_year
    
def conv_incidence(incidenceAngle):
    """
    Return two features describing scaled incidence angle and 
    the wave mode label (0 or 1). Wave mode is 1 if angle is > 30 deg.
    """    
    incidenceAngle[incidenceAngle > 90] = 30
    lbl = np.array(incidenceAngle > 30, dtype='float32')
    incidenceAngle = incidenceAngle / 30.
    return np.column_stack([incidenceAngle, lbl])

def conv_incidence_iw(incidenceAngle):
    # info from esa copernicus website : 29.1° - 46.0° IW
    #min_inc_iw = 29.1
    min_inc_iw = 22 #mini for WV
    max_inc_iw = 46.0
    res = (incidenceAngle - min_inc_iw)/max_inc_iw # range 0-0.52
    #res = (incidenceAngle - min_inc_iw) / (max_inc_iw-min_inc_iw) # a better normalization not yet tested 15 oct 21: range 0-1
    return res
