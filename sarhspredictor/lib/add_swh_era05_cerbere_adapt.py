"""
April 2021
A Grouazel
to add SWH from ECMWF as asked by CCI sea state format
tested with en cwave
"""
import os
import logging
import pdb
import cerbere
from ceraux.model import RegularGriddedModel
from cerbere.model.trajectory import Trajectory
from cerbere.mapper.ecmwf05ncfile import ECMWF05NCFile
from cerbere.mapper.ncfile import NCFile
NAMING_CONVENTION = {
    'swh': 'swh_model',
    }
UNITS = {'m s**-1': 'm s-1'}
STANDARD_NAME = {
    'swh': 'sea_surface_wave_significant_height',
    }

def add_swh_era(feature_file,outpath,linear=False,suffix=None):
    """

    :param feature_file: the file in which you want to append a new field
    :param outpath: (str)
    :param linear: bool
    :param suffix: str
    :return:
    """
    # ERA5 model definition
    model = RegularGriddedModel(
        '/home/ref-ecmwf/ERA5/',
        '%Y/%m/era_5-copernicus__%Y%m%d.nc',
        1,
        0.50,
        'ERA5025NCDataset',
        -90, 0, 720, 361,
        modelfeature='CylindricalGridTimeSeries'
        )
    new = None
    # open feature file
    if new is not None:
        mode = 'r'
    else:
        mode = 'r+'

    # feature = cerbere.open_as_feature(
    #     options.feature, feature_file, options.reader, mode=mode
    # )
    hh = NCFile(feature_file)
    feature = Trajectory.load(hh)

    # remap model fields
    intmode = 'closest'
    if linear:
        intmode = 'linear'
    fieldnames = NAMING_CONVENTION.keys()
    fields = model.get_model_fields(feature, fieldnames, mode=intmode)

    # add metadata
    for fieldname, field in fields.items():

        newname = NAMING_CONVENTION[fieldname]
        field.name = newname
        field.standardname = STANDARD_NAME[fieldname]
        if field.units in UNITS:
            field.units = UNITS[field.units]
        field.attrs['source'] = "Copernicus ERA5 Reanalysis by ECMWF"

    # add to current feature
    if suffix is None:

        for fieldname, field in fields.items():
            if field in feature_file.fieldnames:
                raise Exception('%s field already in file' % fieldname)

            feature.add_field(field)

    # or create a new file with same structure
    else:
        basef, ext = os.path.splitext(
            os.path.basename(feature_file)
            )
        if outpath is None:
            outpath = os.path.dirname(feature_file)
        auxf = os.path.join(
            outpath,
            basef + suffix + '.nc'
            )
        if os.path.exists(auxf):
            os.remove(auxf)
        print("Save to: {}".format(auxf))

        # create empty feature
        auxfeature = feature.extract(fields=[])

        for fieldname, field in fields.items():
            auxfeature.add_field(field)

            # coordinates attribute
            exfield = feature.get_field(feature.fieldnames[0])
            if 'coordinates' in exfield.attrs:
                field.attrs['coordinates'] = \
                    exfield.attrs['coordinates']

        # save
        auxfeature.save(auxf)

    feature.close()
