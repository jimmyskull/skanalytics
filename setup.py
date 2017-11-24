# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='skanalytics',
    version='0.3',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Python Analytics Tools',
    long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='https://github.com/jimmyskull/skanalytics',
    author='Paulo Roberto Urio',
    author_email='paulo@bk.ru'
)
