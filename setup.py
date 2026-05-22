#!/usr/bin/env python
from setuptools import setup, find_packages

with open("mml/__init__.py") as fin:
    for line in fin:
        if line.startswith("__version__ ="):
            version = eval(line[14:])
            break

setup(name='mml',
      version=version,
      description='Multimode learning for verifications in optical computing.',
      author='Zhenyu ZHAO',
      author_email='sjtuzhaozhenyu98@sjtu.edu.cn',
      install_requires=['torch', 'torchvision', 'numpy'],
      url="https://github.com/Hideousmon/multimode-learning",
      packages=find_packages()
      )