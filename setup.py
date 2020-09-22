#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
Topic :: Scientific/Engineering
License :: OSI Approved :: BSD License
Operating System :: OS Independent
Natural Language :: English
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
"""

MAJOR = 0
MINOR = 5
MICRO = 1
RC_INDEX = 0
IS_RELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
FULL_VERSION = '%d.%d.%drc%d' % (MAJOR, MINOR, MICRO, RC_INDEX)


def write_version_py():
    """ This way of version handling is borrowed from the numpy project"""
    cnt = """
# THIS FILE IS GENERATED FROM PYINDUCT SETUP.PY
#
version = '%(version)s'
full_version = '%(full_version)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    filename = os.path.join("pyinduct", "version.py")
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULL_VERSION,
                       'isrelease': str(IS_RELEASED)})
    finally:
        a.close()


def setup_package():
    write_version_py()

    with open('README.rst') as readme_file:
        readme = readme_file.read()

    with open('HISTORY.rst') as history_file:
        history = history_file.read().replace('.. :changelog:', '')

    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    setup(
        name='pyinduct',
        version=VERSION,
        description="Toolbox for control and observer design for "
                    "infinite dimensional systems.",
        long_description=readme[21:],
        long_description_content_type="text/x-rst",
        author="Stefan Ecklebe, Marcus Riesmeier",
        author_email='stefan.ecklebe@tu-dresden.de, marcus.riesmeier@umit.at',
        url='https://github.com/pyinduct/pyinduct/',
        download_url='https://github.com/pyinduct/pyinduct/releases',
        project_urls={
            "Documentation": "https://pyinduct.readthedocs.org",
            "Source Code": "https://github.com/pyinduct/pyinduct/",
            "Bug Tracker": "https://github.com/pyinduct/pyinduct/issues",
        },
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,
        license="BSD 3-Clause License",
        zip_safe=False,
        keywords='distributed-parameter-systems control observer simulation',
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        test_suite='unittest',
        extras_require={
            "tests": ["codecov>=2.0.15"],
            "docs": ["sphinx>=3.2.1",
                     "sphinx-autoapi>=1.5.0",
                     "sphinx-rtd-theme>=0.5.0"],
        },
        python_requires=">=3.5",
    )


if __name__ == "__main__":
    setup_package()
