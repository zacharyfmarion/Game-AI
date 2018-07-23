# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

long_description = '''
GameAI attempts to make it extremely easy to test out different
agents with various board games. It also provides baseline 
implementations of different AI and RL algorithms for community use.
'''

setup(
  name='gameai',
  version='0.1.0',
  description='Set of classes for playing games with agents',
  long_description=long_description,
  author='Zachary Marion',
  author_email='zacharyfmarion@gmail.com',
  url='https://github.com/zacharyfmarion/Game-AI',
  license='MIT',
  install_requires=['tqdm>=4.23.4'],
  packages=find_packages(exclude=('tests', 'docs'))
)