import os
from setuptools import setup

# Change working directory to the one this file is in
os.chdir(os.path.dirname(os.path.realpath(__file__)))

setup(
    name='design_search',
    version='0.1.0',
    packages=['design_search'],
    package_dir={'design_search': '..'},
    install_requires=[
        'numpy >= 1.19'
    ]
)
