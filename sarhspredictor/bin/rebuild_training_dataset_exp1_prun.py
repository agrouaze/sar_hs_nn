# coding: utf-8
import os
import sys
import subprocess
import logging

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
    # see python /home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/bin/extract_list_days_in_cwaveV4_for_parallel_processing.py --outputdir /home1/datawork/agrouaze/data/sentinel1/cwave/
    listing = '/home1/datawork/agrouaze/data/sentinel1/cwave/listing_SAR_L2_L1_measu_from_colocations_cwaveV4_consolidated.txt'
    #listing = '/home1/datawork/agrouaze/data/sentinel1/cwave/listing_SAR_L2_L1_measu_from_colocations_cwaveV4_consolidated_2015.txt'
    listing = '/home1/datawork/agrouaze/data/sentinel1/cwave/listing_SAR_L2_L1_measu_from_colocations_cwaveV4_consolidated_2018.txt'
    # call prun
    opts = ' --split-max-lines=75 --name exp1NN --background -e '
    opts = ' --split-max-lines=5 --name exp1NN --background -e ' # pour 2015
    opts = ' --split-max-lines=18 --name exp1NN --background -e ' # pour 2018 178086 tiff SLC colocalised
    pbs = '/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/bin/rebuild_training_dataset_exp1.pbs'
    cmd = prunexe+opts+pbs+' '+listing
    logging.info('cmd to cast = %s',cmd)
    st = subprocess.check_call(cmd,shell=True)
    logging.info('status cmd = %s',st)