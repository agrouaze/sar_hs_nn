"""
14 June 2021
context for CCI format we need the params meteo from ERA05
inspired from https://gitlab.ifremer.fr/cciseastate/cciseastate/-/blob/master/bin/add_era5
"""
import logging
import xarray
import datetime
import os
import pdb
import time
import numpy as np
from affine import Affine
NAMING_CONVENTION = {
    'u10': 'wind_speed_model_u',
    'v10': 'wind_speed_model_v',
    'sst': 'sea_surface_temperature',
    't2m': 'surface_air_temperature',
   # 'tccw': 'total_liquid_water_content',
    'msl': 'surface_air_pressure'
    }
UNITS = {'m s**-1': 'm s-1'}
STANDARD_NAME = {
    'u10': 'eastward_wind',
    'v10': 'northward_wind',
    'sst': 'sea_surface_temperature',
    't2m': 'air_temperature',
    'msl': 'atmosphere_cloud_liquid_water_content',
    #'tccw': 'atmosphere_mass_content_of_cloud_liquid_water'
}

def get_params_at_locations(lons,lats,dates):
    """

    :param lons:
    :param lats:
    :param dates:
    :return:
    """
    pattern = os.path.join('/home/ref-ecmwf/ERA5/','%s/%s/era_5-copernicus__%s.nc')
    opened_files = {}
    subds = None
    values_coloc = {}
    values_coloc['time'] = []
    list_files_concerned = []
    for ii in range(len(dates)):
        path_filled = pattern%(dates[ii].strftime('%Y'),dates[ii].strftime('%m'),dates[ii].strftime('%Y%m%d'))
        logging.debug('path_filled : %s',path_filled)
        if path_filled not in list_files_concerned:
            list_files_concerned.append(path_filled)
    # if path_filled not in opened_files:
    #     ds = xarray.open_dataset(path_filled)
    #     opened_files[path_filled] = ds
    # else:
    #     ds = opened_files[path_filled]
    logging.info('open %s file(s) ',len(list_files_concerned))
    ds = xarray.open_mfdataset(list_files_concerned,combine='nested',concat_dim='time')
    logging.debug('ds : %s',ds)
    ind_times = []
    for ii in range(len(dates)) :
        date_x = dates[ii]
        toto = abs(ds['time'].values-np.datetime64(date_x))
        ind_closest = np.argmin(toto)
        ind_times.append(ind_closest)
        values_coloc['time'].append(date_x)
    # affine from lon/lat to x/y in output raster (inspired from density_map_golf_du_lion.ipynb)
    resolution = 0.25
    bounds = (0,-90,360.,90.) # version lon lat
    logging.debug('bounds: %s',bounds[0 :2])
    transform = Affine.translation(bounds[0],bounds[1]) * Affine.scale(resolution,resolution)
    indice_max_lat = 721
    indice_max_lon = 1440
    _,lat = transform * (0,indice_max_lat)
    lon,_ = transform * (indice_max_lon,0)
    assert lat==90.25
    # transformation lon,lat -> x,y , on utilise le tilde
    x,y = ~transform * (360,90)
    lons += 180.
    xs,ys = ~transform *(lons,lats)
    xs = np.round(xs).astype(int)
    ys = np.round(ys).astype(int)

    for vv in STANDARD_NAME:
        if vv not in values_coloc:
            values_coloc[vv] = ds[vv].values[ind_times,ys,xs]
        else:
            values_coloc[vv] = np.hstack([values_coloc[vv],ds[vv].values[ind_times,ys,xs]])

    logging.info('declare dataset')
    subds = xarray.Dataset()
    for vv in values_coloc:
        if vv not in ['time']:
            subds[vv] = xarray.DataArray(values_coloc[vv],dims=['times'],coords={'times':values_coloc['time']})
            #for aatk in ['long_name','standard_name']:
            subds[vv].attrs['long_name'] = STANDARD_NAME[vv]
            subds[vv].attrs['standard_name'] = NAMING_CONVENTION[vv]

    return subds

def from_np64_to_dt(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)

if __name__ == '__main__' :
    tinit = time.time()
    root = logging.getLogger()
    if root.handlers :
        for handler in root.handlers :
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='hs_sar_product')
    parser.add_argument('--verbose',action='store_true',default=False)
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')

    testfile = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/L2_daily/0.8/2021/001/S1A_WV_L2D_enriched_LOPS_20210101_daily_IPF_003.31.nc'
    tmpds = xarray.open_dataset(testfile)
    logging.debug('tmpds: %s',tmpds)
    lons = tmpds['lon'].values
    lats = tmpds['lat'].values
    dates = np.array([from_np64_to_dt(xx) for xx in tmpds['fdatedt'].values])
    logging.debug('nb locations: %s',len(lons))
    logging.debug('dates: %s %s',dates,dates.dtype)
    dsera05 = get_params_at_locations(lons,lats,dates)
    logging.info('dsera05 : %s',dsera05)
    pdb.set_trace()