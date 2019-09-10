#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('test_requirements.txt') as f:
    test_requirements = f.read().splitlines()


setup(
    name='pyinduct',
    version='0.5.0',
    description="The aim of this project is to provide a toolbox to automate "
                "the backstepping based design of controllers for boundary "
                "actuated infinite dimensional systems.",
    long_description=readme + '\n\n' + history,
    author="Stefan Ecklebe, Marcus Riesmeier",
    author_email='stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at',
    url='https://github.com/pyinduct/pyinduct/',
    packages=[
        'pyinduct',
    ],
    package_dir={'pyinduct':
                 'pyinduct'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD 3-Clause License",
    zip_safe=False,
    keywords='pyinduct',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='pyinduct.tests',
    tests_require=test_requirements
)
