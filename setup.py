# /path/to/STARE/setup.py
from setuptools import setup, find_packages

setup(
    name='stare_lib',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)