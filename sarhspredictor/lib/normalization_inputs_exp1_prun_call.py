#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
# coding: utf-8
"""
temps estimated of run from 2014 to 2018: about 713k files WV: about 30min
"""
import os
import sys
import subprocess
import logging
from dateutil import rrule
import datetime
import numpy as np
if __name__ == '__main__':
    root = logging.getLogger ()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler (handler)
    import argparse
    parser = argparse.ArgumentParser (description='start prun')
    parser.add_argument ('--verbose',action='store_true',default=False)
    args = parser.parse_args ()
    if args.verbose:
        logging.basicConfig (level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig (level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    prunexe = '/appli/prun/bin/prun'
    #listing = '/home1/scratch/agrouaze/agg_multi_year_sat_prun.txt'
    listing = '/home1/scratch/agrouaze/hs_regression_listing_exp1_normalization_step_daily_prun.txt' # written below
    listing = '/home1/scratch/agrouaze/hs_regression_listing_exp2_normalization_step_daily_prun.txt'  # written below
    # call prun
    opts = ' --split-max-lines=1 --name exp1HsNormDailyProd --background -e ' # pour 2018 178086 tiff SLC colocalised
    listing_content = []
    #sta = datetime.datetime(2014,4,1)
    #sto = datetime.datetime.today()
    #sto  = datetime.datetime(2018,12,31)
    sta = datetime.datetime(2019,7,1)
    sto = datetime.datetime(2022,7,4)
    for ddd in rrule.rrule(rrule.DAILY,dtstart=sta,until=sto):
            #listing_content.append('--startdate %s0101 --stopdate %s1231 --suffix=%s'%(year,year,year))
            listing_content.append('%s %s' % (ddd.strftime('%Y%m%d'),ddd.strftime('%Y%m%d')))
    fid  = open(listing,'w')
    for uu in listing_content:
        fid.write(uu+'\n')
    fid.close()
    logging.info('listing written ; %s',listing)
    #pbs = '/home1/datahome/agrouaze/git/mpc/data_collect/aggregate_daily_s1wvocn_production_prun.pbs' # this one is the previous with lot of memory reservation
    #pbs = '/home1/datahome/agrouaze/git/mpc/data_collect/aggregate_daily_s1wvocn_production_splited_by_year_and_sat.pbs'
    pbs = '/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/lib/normalization_inputs_exp1.pbs'
    cmd = prunexe+opts+pbs+' '+listing
    logging.info('cmd to cast = %s',cmd)
    st = subprocess.check_call(cmd,shell=True)
    logging.info('status cmd = %s',st)