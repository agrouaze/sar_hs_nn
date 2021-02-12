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
    'long_description': "# sar_hs_nn lib and bin to predict Hs and Hs_std from SAR Sentinel-1 mission in WV. Based on Quach 2020 model.Ifremer implementation 2021.",
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
