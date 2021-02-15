# sar_hs_nn
With this python library you can:
 1) predict Hs and Hs_std from SAR Sentinel-1 mission in WV.
 2) compute the 20 C-WAVE params.
 3) generate netCDF file in the CCI Sea state official format.
 
Based on Quach 2020 model: https://authors.library.caltech.edu/104562/1/09143500.pdf 

Ifremer implementation 2021.
Used to produce CCI SeaState SAR product.

`sarhs` lib is a copy paste from https://github.com/hawaii-ai/SAR-Wave-Height .

 # Installation of the lib in a conda environement

1) install conda
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
2) create a conda env
 ```
 conda create -n sarhs python=3.7.8
 conda activate sarhs
```

3) clone the git repository.
```bash
git clone https://github.com/grouny/sar_hs_nn.git
cd sar_hs_nn
```

4) install the `sarhspredictor` and its dependencies
```bash
python setup.py install
```
if the previous command failed try to run the command in error by yourself. It should looks like:
```bash
../bin/python3.7 -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'../sar_hs_nn/setup.py'"'"'; __file__='"'"'../sar_hs_nn/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' develop --no-deps
```

5) check installation
```python
import sarhspredictor
```

6) edit the `sarhspredictor/config.py` file
 In this file you can set the path of the different models .h5

# usage
to predict a Hs from WV Sentinel-1 SAR Level-2 product, follow this [python notebook demo](https://github.com/grouny/sar_hs_nn/blob/main/sarhspredictor/examples/predict_Hs_using_quach2020_model_from_S1_WV_OCN_files.ipynb):

to prepare the training dataset using the colocations files bteween S-1 and altimeters:
```bash
python sarhspreidctor/bin/rebuild_training_dataset.py --input /home/cercache/users/jstopa/sar/empHs/cwaveV4/S1A_ALT_coloc201501S.nc
```

to create a netCDF file containing the predictions starting from the reference inputs (J. Stopa nc file):
```bash
python sarhspredictor/bin/predict_with_quach2020_from_ref_input_using_keras.py --modelversion heteroskedastic_2017.h5
```
to create a netCDF file containing the predictions starting from ESA S-1 Level-2 WV: 
```bash
python sarhspredictor/bin/predict_and_save_nc_from_OCN_using_keras_based_on_ref_listing_files.py --modelversion heteroskedastic_2017.h5
```
to create a netCDF file in CCI sea state format containing the Hs predicted from WV OCN files:

```bash
usage: generate_cci_sea_state_daily_nc_file.py [-h] [--verbose]
                                               [--outputdir OUTPUTDIR]
                                               --startdate STARTDATE
                                               --stopdate STOPDATE --sat SAT
                                               --wv WV [--redo]
                                               --cwave-version CWAVE_VERSION
                                               [--dev]

hs_sar_product

optional arguments:
  -h, --help            show this help message and exit
  --verbose
  --outputdir OUTPUTDIR
                        folder where the data will be written [optional]
  --startdate STARTDATE
                        YYYYMMDD
  --stopdate STOPDATE   YYYYMMDD
  --sat SAT             S1A or S1B...
  --wv WV               wv1 or wv2...
  --redo                redo existing files nc
  --cwave-version CWAVE_VERSION
                        example v1.2
  --dev                 dev/test mode only 2 wv measu treated in a day
```