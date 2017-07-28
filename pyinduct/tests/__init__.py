# -*- coding: utf-8 -*-
import sys
import os

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    test_all_examples = False
    show_plots = False
elif any(["pyinduct{}tests{}".format(os.sep, os.sep) in arg for arg in sys.argv]):
    test_all_examples = True
    show_plots = False
else:
    test_all_examples = False
    show_plots = True
    # Do not want to see plots while test run?
    # Then force it and uncomment the next line!
    # show_plots = False
