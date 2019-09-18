#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


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
    version='0.5.0experimental-symbolic',
    description="Toolbox for control and observer design for "
                "infinite dimensional systems.",
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="Stefan Ecklebe, Marcus Riesmeier",
    author_email='stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at',
    url='https://github.com/pyinduct/pyinduct/',
    download_url='https://github.com/pyinduct/pyinduct/releases',
    project_urls={
        "Documentation": "https://pyinduct.readthedocs.or",
        "Source Code": "https://github.com/pyinduct/pyinduct/",
        "Bug Tracker": "https://github.com/pyinduct/pyinduct/issues",
    },
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="BSD 3-Clause License",
    zip_safe=False,
    keywords='distributed-parameter-systems control observer simulation',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='pyinduct.tests',
    tests_require=test_requirements,
    python_requires=">=3.5",
)
