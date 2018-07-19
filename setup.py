# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.rst') as f:
  readme = f.read()

with open('LICENSE') as f:
  license = f.read()

setup(
  name='gameai',
  version='0.1.0',
  description='Set of classes for playing games with agents',
  long_description=readme,
  author='Zachary Marion',
  author_email='zacharyfmarion@gmail.com',
  url='https://github.com/zacharyfmarion/Game-AI',
  license=license,
  packages=find_packages(exclude=('tests', 'docs'))
)