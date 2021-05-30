"""
exp 1: doing the Quach experiment with X spectra from SLC instead of OCN
May 2021
A Grouazel
the listing of tiff to treat is here: /home1/datawork/agrouaze/data/sentinel1/cwave/listing_SAR_L2_L1_measu_from_colocations_cwaveV4.txt
inspiration rebuild_training_dataset.py
"""
def prepare_one_measurement(slc,ocn):
    """

    :param slc:
    :param ocn:
    :return:
    """
    # 1) read all params from Justin s dataset
    # 2) read X spectra from tiff
    # 3) interpolate and convert cartesian grid to polar 72,60
    # 4) compute C wave params
    # save a netcdf file
