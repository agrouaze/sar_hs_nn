__all__ = (
    '__version__',
    'sarhspredictor'
)

# Or substitute a different method and assign the result to __version__
import pkg_resources  # part of setuptools


# This one seems to be slower, and with pyinstaller makes the exe a lot bigger
try:
    from pbr.version import VersionInfo
    v3 = VersionInfo('sarhspredictor').release_string()
except:
    v3='unknown_need_install_via_pip'
#print('v3 {}'.format(v3))
#__version__ = pkg_resources.get_distribution("sarhspredictor").version
__version__ = v3

#import importlib.metadata

#__version__ = importlib.metadata.version('sarhspredictor')