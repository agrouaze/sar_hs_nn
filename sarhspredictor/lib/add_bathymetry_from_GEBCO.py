"""

inspired from ceraux code
"""
import numpy
import logging
from netCDF4 import Dataset


def decdeg2dms ( dd ) :
    minutes,seconds = divmod(numpy.abs(dd) * 3600,60)
    degrees,minutes = divmod(minutes,60)

    if isinstance(dd,numpy.ndarray) :
        degrees[((dd < 0) & (degrees > 0))] *= -1.
        minutes[((dd < 0) & (degrees <= 0) & (minutes > 0))] *= 1.
        seconds[((dd < 0) & (degrees <= 0) & (minutes <= 0))] *= 1.

    else :
        if dd < 0 :
            if degrees > 0 :
                degrees = -degrees
            elif minutes > 0 :
                minutes = -minutes
            else :
                seconds = -seconds

    return (degrees,minutes,seconds)


class GEBCO14Bathymetry() :
    """A class to handle GEBCO bathymetry information."""

    RES_30S = '30s'
    RES_1M = '1min'

    def __init__ ( self,filename,cache=False ) :
        self._bathyfile = Dataset(filename,'r')
        self.cache = cache
        if cache :
            self.bathymetry = self._bathyfile.variables['z'][:]
        else :
            self.bathymetry = self._bathyfile.variables['z']

        if len(self._bathyfile.dimensions['xysize']) == 933120000 :
            self._resolution = GEBCO14Bathymetry.RES_30S
            self._factor = 2.
            self._offset = 43200
        else :
            self._resolution = GEBCO14Bathymetry.RES_1M
            self._factor = 1
            self._offset = 21601

    def __del__ ( self ) :
        if self._bathyfile is not None :
            self._bathyfile.close()

    def get_depth ( self,lon,lat ) :
        """Return the depth, in meters"""
        deg,minu,sec = decdeg2dms(lon)
        lon_in_min = deg * 60 + minu + sec / 60.

        deg,minu,sec = decdeg2dms(lat)
        lat_in_min = deg * 60 + minu + sec / 60.

        indices = (
                (180. * 60 * self._factor + numpy.floor(lon_in_min * self._factor)
                 ).astype(numpy.int)
                + self._offset * (
                        90. * 60. * self._factor - numpy.floor(lat_in_min * self._factor)
                ).astype(numpy.int)
        )

        if self.cache :
            return self.bathymetry[indices]
        else :
            if not isinstance(indices,numpy.ndarray) :
                return self.bathymetry[indices]
            return numpy.ma.array(
                [self.bathymetry[_] for _ in indices]
            )

    # def _get_depth_field_attributes ( self ) :
    #     attrs = super(GEBCO14Bathymetry,self)._get_depth_field_attributes()
    #     attrs['source'] = "The GEBCO_2014 Grid, version 20150318, www.gebco.net,  doi:10.1002/2015EA000107"
    #     attrs['institution'] = "IOC/IHO"
    #     return attrs