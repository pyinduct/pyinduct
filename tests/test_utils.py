import os
import sys
import unittest

import pyinduct as pi
from pyinduct import utils as ut

if any([arg in {'discover', 'setup.py', 'test'} for arg in sys.argv]):
    show_plots = False
else:
    show_plots = True
    # show_plots = False

if show_plots:
    import pyqtgraph as pg

    app = pg.QtGui.QApplication([])




class CreateDirTestCase(unittest.TestCase):
    existing_file = "already_a_file_there"
    existing_dir = "already_there"
    new_dir = "not_yet_there"

    def setUp(self):
        # check if test directories already exist and stop if they do
        for name in [self.existing_dir, self.new_dir]:
            dir_name = os.sep.join([os.getcwd(), name])
            if os.path.exists(dir_name):
                self.fail("test directory already exists, tests cannot be run.")

    def test_existing_file(self):
        # create a file with directory name
        dir_name = os.sep.join([os.getcwd(), self.existing_dir])
        with open(dir_name, "w") as f:
            pass

        self.assertRaises(FileExistsError, ut.create_dir, self.existing_dir)
        os.remove(dir_name)

    def test_existing_dir(self):
        dir_name = os.sep.join([os.getcwd(), self.existing_dir])
        os.makedirs(dir_name)
        ret = ut.create_dir(self.existing_dir)
        self.assertTrue(os.path.exists(dir_name))  # do not remove the directory
        self.assertEqual(ret, dir_name)  # return abs path of created dir
        os.rmdir(dir_name)

    def test_non_existing_dir(self):
        dir_name = os.sep.join([os.getcwd(), self.new_dir])
        ret = ut.create_dir(self.new_dir)
        self.assertTrue(os.path.exists(dir_name))  # directory should be created
        self.assertEqual(ret, dir_name)  # return abs path of created dir

        os.rmdir(dir_name)


class CreateVideoTestCase(unittest.TestCase):
    @unittest.skip("unfinished test case that requires ffmpeg")
    def test_creation(self):
        # TODO generate test data first!
        ut.create_animation("./animation_output/Test_Plot_21_55_32_%03d_.png")
