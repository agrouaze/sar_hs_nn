# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['sarhspredictor',
 'sarhspredictor.bin',
 'sarhspredictor.lib',
 'sarhspredictor.lib.sarhs']

package_data = \
{'': ['*']}

install_requires = \
['click>=7.1,<8.0',
 'jinja2>=2,<3',
 'keras-applications==1.0.8',
 'keras-preprocessing==1.1.0',
 'keras==2.3.1',
 'loguru==0.5',
 'netcdf4==1.5.4',
 'numpy==1.19.2',
 'poetry==1.1.4',
 'scipy==1.5.2',
 'setuptools',
 'sherpa==4.12.1',
 'tensorflow>=1.15.0',
 'xarray==0.16.2']

entry_points = \
{'console_scripts': ['greet = python_bootstrap.greeting.cli:main']}

setup_kwargs = {
    'name': 'sarhspredictor',
    'version': '0.0.0',
    'description': 'Hs prediction from SAR',
    'long_description': "# sar_hs_nn\nlib and bin to predict Hs and Hs_std from SAR Sentinel-1 mission in WV.\nBased on Quach 2020 model.\nIfremer implementation 2021.\nUsed to produce CCI SeaState SAR product.\n\nsarhs lib is a copy paste from https://github.com/hawaii-ai/SAR-Wave-Height .\n\n # Installation\n install `poetry` binary\n ```bash\ncurl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -\n```\n create a conda env\n ```\n conda create -n sarhs python=3.7.8\n conda activate sarhs\n```\nadd setuptools\n ```bash\nconda install -c anaconda setuptools\n```\ncreate a setupy.py with `poetry`\n```bash\n$HOME/.poetry/bin/poetry build --format sdist && tar --wildcards -xvf dist/*.tar.gz -O '*/setup.py' > setup.py\n```\nclone the git repository.\n```bash\ngit clone https://github.com/grouny/sar_hs_nn.git\ncd sar_hs_nn\n```\n\ninstall the sarhspredictor and its dependencies\n```bash\npip install -e .\n```\n\ncheck installation\n```python\nimport sarhspredictor\n```\n",
    'author': 'Antoine GROUAZEL',
    'author_email': 'antoine.grouazel@ifremer.fr',
    'maintainer': 'Antoine GROUAZEL',
    'maintainer_email': 'antoine.grouazel@ifremer.fr',
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.7,<4',
}


setup(**setup_kwargs)
