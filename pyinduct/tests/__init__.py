# -*- coding: utf-8 -*-
import sys
import warnings
import numpy as np

# since this is a serious toolbox
warnings.warn("Test Mode: Treating all numerical warnings as errors")
np.seterr(all="raise")

test_examples = None
test_timings = None
show_plots = None
if any([arg in {"discover"} for arg in sys.argv]):
    warnings.warn("Operating in global test mode, no windows will be shown.")
    # global run of all tests
    test_examples = True
    test_timings = True
    show_plots = False
elif any(['sphinx-build' in arg for arg in sys.argv]):
    # sphinx build run
    test_examples = False
    test_timings = False
    show_plots = False
else:
    # local run
    test_examples = True
    test_timings = True
    show_plots = True

# Do not want to see plots or test all examples while test run?
# Then force it and uncomment the respective line:
# test_timings = False
# test_all_examples = False
# show_plots = False
