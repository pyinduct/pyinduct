import sys
import os
import unittest

if __name__ == '__main__':
    """
    Standard hook to tun all testing
    """
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), "tests")
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner(buffer=True)
    runner.run(suite)
