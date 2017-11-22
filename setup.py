#!/usr/bin/env python
# -*- coding: utf-8 -*-

# try:
from setuptools import setup, find_packages
import os
# except ImportError:
#     from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='pyinduct',
    version='0.4.0',
    description="The aim of this project is to provide a toolbox to automate the backstepping based design of "
                "controllers for boundary actuated infinite dimensional systems.",
    long_description=readme + '\n\n' + history,
    author="Stefan Ecklebe, Marcus Riesmeier",
    author_email='stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at',
    url='https://github.com/pyinduct/pyinduct/',
    # packages=find_packages(),
    packages=[
        'pyinduct',
    ],
    package_dir={'pyinduct':
                 'pyinduct'},
    include_package_data=True,
    install_requires=requirements,
    license="GPL v3",
    zip_safe=False,
    keywords='pyinduct',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        # "Programming Language :: Python :: 2",
        # 'Programming Language :: Python :: 2.6',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='pyinduct.tests',
    tests_require=test_requirements
)
