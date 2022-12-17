import os
from setuptools import setup

# Change working directory to the one this file is in
os.chdir(os.path.dirname(os.path.realpath(__file__)))

setup(
    name='${LIBRARY_NAME}',
    version='0.1.0',
    packages=['${LIBRARY_NAME}'],
    package_dir={'${LIBRARY_NAME}': '..'}
)
