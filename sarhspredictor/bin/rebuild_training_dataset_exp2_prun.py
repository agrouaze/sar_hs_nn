#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
# coding: utf-8
"""
June 2022
Grouazel
temps total a priori
995 jours pour les CMEMS
2 SAR
3 alti
2h environs pour une journée
et 95 jobs en parallel
ca fait du 125heures total -> 5,2 jours.
"""
import datetime

import os
import sys
import subprocess
from dateutil import rrule
import logging
POSSIBLES_CMEMS_ALTI = {'cryosat-2':'c2',
                      #'envisat':'ENVISAT',
                     #'jason-1':'Jason-3',
                    'cfosat':'cfo',
                     'jason-2':'j2',
                     'jason-3':'j3',
                     'saral':'al',
                    'sentinel-3a':'s3a',
                    'sentinel-3b':'s3b',
                        } # in v2.0.6
if __name__ == '__main__':
    root = logging.getLogger ()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler (handler)
    import argparse

    parser = argparse.ArgumentParser (description='start prun')
    parser.add_argument ('--verbose',action='store_true',default=False)
    parser.add_argument('--onlymissing', action='store_true', default=False,help='filter inputs listing to process only missing dates')
    args = parser.parse_args ()

    if args.verbose:
        logging.basicConfig (level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig (level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    prunexe = '/appli/prun/bin/prun'
    #outputdir = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp2D4/v1/'
    outputdir = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/hs_nn/exp2D4/v2/' ##30 June 2022 add azimuth cutoff, and more colocations +20% environ
    #sta = datetime.datetime(2019,10,1)
    # les produits dataset-wav-alti-l3-swh-rt-global-cfo commencent avant les autres
    sta = datetime.datetime(2019, 7,16)
    sto = datetime.datetime.today()

    for alti in POSSIBLES_CMEMS_ALTI:
        for sarunit in ['S1A','S1B']:
            listing = '/home1/datawork/agrouaze/data/sentinel1/cwave/listing_Hs_regression_exp2_analog_%s_%s.txt'%(alti,sarunit)
            fid = open(listing,'w')
            cpt = 0
            cpt_already_present = 0
            for dd in rrule.rrule(rrule.DAILY,dtstart=sta,until=sto):
                pot_output = os.path.join(outputdir,dd.strftime('%Y'),
                              dd.strftime('%j'),'training_D4_exp2_%s_%s_%s.nc' %(dd.strftime('%Y%m%d'),alti,sarunit))
                if os.path.exists(pot_output):
                    cpt_already_present+=1
                if os.path.exists(pot_output) and args.onlymissing:
                    pass
                else:
                    fid.write('%s %s %s %s \n'%(dd.strftime('%Y%m%d'),sarunit,alti,outputdir))
                cpt +=1
            fid.close()
            logging.info('listing %s lines %s cpt_already_present %s',listing,cpt,cpt_already_present)
            # call prun
            # opts = ' --split-max-lines=75 --name exp1NN --background -e '
            # opts = ' --split-max-lines=5 --name exp1NN --background -e ' # pour 2015
            # #opts = ' --split-max-lines=18 --name exp1NN --background -e ' # pour 2018 178086 tiff SLC colocalised
            # opts = ' --split-max-lines=25 --name exp1NN --background -e '  # pour 2016 or 2017 242897 tiff/ocn
            opts = ' --split-max-jobs=999 --name exp2%s_%s --background -e '%(alti,sarunit)  # pour full listing ~71 WV images par sous job
            pbs = '/home1/datahome/agrouaze/git/sar_hs_nn/sarhspredictor/bin/rebuild_training_dataset_exp2.pbs'
            cmd = prunexe+opts+pbs+' '+listing
            logging.info('cmd to cast = %s',cmd)
            st = subprocess.check_call(cmd,shell=True)
            logging.info('status cmd = %s',st)
            #stop
