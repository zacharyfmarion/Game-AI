# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setup(
    name='gameai',
    version='0.3.2',
    description='Set of classes for playing games with agents',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Zachary Marion',
    author_email='zacharyfmarion@gmail.com',
    url='https://github.com/zacharyfmarion/Game-AI',
    license='MIT',
    install_requires=['tqdm>=4.23.4'],
    extras_require={
        'tests': [
            'pytest >= 4.0.2'],
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme']},
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)
