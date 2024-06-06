from setuptools import find_packages
from setuptools import setup

setup(
    name='asitlorbot8_firmware',
    version='0.0.0',
    packages=find_packages(
        include=('asitlorbot8_firmware', 'asitlorbot8_firmware.*')),
)
