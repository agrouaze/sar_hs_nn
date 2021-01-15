# sar_hs_nn
lib and bin to predict Hs and Hs_std from SAR Sentinel-1 mission in WV.
Based on Quach 2020 model.
Ifremer implementation 2021.
Used to produce CCI SeaState SAR product.

sarhs lib is a copy paste from https://github.com/hawaii-ai/SAR-Wave-Height .

 # Installation
 install `poetry` binary
 ```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
 create a conda env
 ```
 conda create -n sarhs python=3.7.8
 conda activate sarhs
```
add setuptools
 ```bash
conda install -c anaconda setuptools
```
create a setupy.py with `poetry`
```bash
$HOME/.poetry/bin/poetry build --format sdist && tar --wildcards -xvf dist/*.tar.gz -O '*/setup.py' > setup.py
```
clone the git repository.
```bash
git clone https://github.com/grouny/sar_hs_nn.git
cd sar_hs_nn
```

install the sarhspredictor and its dependencies
```bash
pip install -e .
```

check installation
```python
import sarhspredictor
```
