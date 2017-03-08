# -*- coding: utf-8 -*-
import sys

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    # Do not want to see plots while test run?
    # Then force it and uncomment the next line!
    # show_plots = False
