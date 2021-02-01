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
1) install `poetry` binary
 ```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
2) create a conda env
 ```
 conda create -n sarhs python=3.7.8
 conda activate sarhs
```
3) add setuptools
 ```bash
conda install -c anaconda setuptools
```
5) clone the git repository.
```bash
git clone https://github.com/grouny/sar_hs_nn.git
cd sar_hs_nn
```

4) create a setupy.py with `poetry`
```bash
$HOME/.poetry/bin/poetry build --format sdist && tar --wildcards -xvf dist/*.tar.gz -O '*/setup.py' > setup.py
```


install the `sarhspredictor` and its dependencies
```bash
pip install -e .
```
if the previous command failed try to run the command in error by yourself. It should looks like:
```bash
../bin/python3.7 -c 'import sys, setuptools, tokenize; sys.argv[0] = '"'"'../sar_hs_nn/setup.py'"'"'; __file__='"'"'../sar_hs_nn/setup.py'"'"';f=getattr(tokenize, '"'"'open'"'"', open)(__file__);code=f.read().replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' develop --no-deps
```

check installation
```python
import sarhspredictor
```

edit the `sarhspredictor/config.py` file in order to set the path of the different models .h5

# usage
to predict a Hs from WV Sentinel-1 SAR Level-2 product, follow this [python notebook demo](https://github.com/grouny/sar_hs_nn/blob/main/sarhspredictor/examples/predict_Hs_using_quach2020_model_from_S1_WV_OCN_files.ipynb):


to create a netCDF file containing the predictions starting from the reference inputs (J. Stopa nc file):
```bash
python sarhspredictor/bin/predict_with_quach2020_from_ref_input_using_keras.py --modelversion heteroskedastic_2017.h5
```
to create a netCDF file containing the predictions starting from ESA S-1 Level-2 WV: 
```bash
python sarhspredictor/bin/predict_and_save_nc_from_OCN_using_keras_based_on_ref_listing_files.py --modelversion heteroskedastic_2017.h5
```
