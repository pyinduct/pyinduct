# -*- coding: utf-8 -*-
import sys

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    test_examples = True
    test_timings = True
    test_all_examples = False
    show_plots = False
elif any(['sphinx-build' in arg for arg in sys.argv]):
    test_examples = False
    test_timings = False
    test_all_examples = False
    show_plots = False
else:
    test_examples = True
    test_timings = True
    test_all_examples = True
    show_plots = True
    # Do not want to see plots or test all examples while test run?
    # Then force it and uncomment the respective line:
    test_timings = False
    # test_all_examples = False
    show_plots = False
