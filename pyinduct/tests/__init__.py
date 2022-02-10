# -*- coding: utf-8 -*-
import sys
import os
import warnings
import numpy as np

__all__ = ["get_test_resource_path"]


# the default is to run all tests and to treat numerical warnings as errors
test_examples = True
test_timings = True
show_plots = True

if any([arg in {"--no-plots"} for arg in sys.argv]):
    warnings.warn("No plots will be shown.")
    show_plots = False
if any([arg in {"--no-examples"} for arg in sys.argv]):
    warnings.warn("Tests on examples will not be run")
    test_examples = False
if any([arg in {"--no-timings"} for arg in sys.argv]):
    warnings.warn("Tests on timings will not be run")
    test_timings = False
if not any([arg in {"--no-np-strict-mode"} for arg in sys.argv]):
    warnings.warn("Test Mode: Treating all numerical warnings as errors")
    np.seterr(all="raise")

# Do not want to see plots or test all examples while test run?
# Then force it and uncomment the respective line:
# test_timings = False
# test_all_examples = False
# show_plots = False


def get_test_resource_path(res_name):
    """
    Utility to get the absolute path for a test resource
    Args:
        res_name(str): Name of the resource file.

    Returns: Absolute path if the resource.
    """
    own_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(own_path, "test_data", res_name)
    return file_path
