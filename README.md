# sar_hs_nn
lib and bin to predict Hs and Hs_std from SAR Sentinel-1 mission in WV.
Based on Quach 2020 model.
Ifremer implementation 2021.
Used to produce CCI SeaState SAR product.

sarhs lib is a copy paste from https://github.com/hawaii-ai/SAR-Wave-Height .

 # Installation
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


install the sarhspredictor and its dependencies
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

edit the sarhspredictor/config.py file in order to set the path of the different models .h5

# usage
to create a netCDF file containing the predictions starting from the reference inputs (J. Stopa nc file):
```python
python sarhspredictor/bin/predict_with_quach2020_from_ref_input_using_keras.py --modelversion heteroskedastic_2017.h5
```
to create a netCDF file containing the predictions starting from ESA S-1 Level-2 WV: 
```python
python sarhspredictor/bin/predict_and_save_nc_from_OCN_using_keras_based_on_ref_listing_files.py --modelversion heteroskedastic_2017.h5
```
