__all__ = (
    '__version__',
    'sarhspredictor'
)

# Or substitute a different method and assign the result to __version__
import pkg_resources  # part of setuptools
from pbr.version import VersionInfo

# This one seems to be slower, and with pyinstaller makes the exe a lot bigger
v3 = VersionInfo('sarhspredictor').release_string()
#print('v3 {}'.format(v3))
#__version__ = pkg_resources.get_distribution("sarhspredictor").version
__version__ = v3

#import importlib.metadata

#__version__ = importlib.metadata.version('sarhspredictor')