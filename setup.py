# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name='gameai',
    version='0.1.0',
    description='Set of classes for playing games with agents',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Zachary Marion',
    author_email='zacharyfmarion@gmail.com',
    url='https://github.com/zacharyfmarion/Game-AI',
    license='MIT',
    install_requires=['tqdm>=4.23.4'],
    packages=find_packages(exclude=('tests', 'docs'))
)
