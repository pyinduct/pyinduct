"""
A few helper functions for users and developers.
"""
import os
from subprocess import call


def create_dir(dir_name):
    """
    Create a directory with name :py:obj:`dir_name` relative to the current
    path if it doesn't already exist and return its full path.

    Args:
        dir_name (str): Directory name.

    Return:
        str: Full absolute path of the created directory.
    """
    path = os.sep.join([os.getcwd(), dir_name])
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise FileExistsError("cannot create directory, file of same name already present.")

    return path


def create_animation(input_file_mask="", input_file_names=None, target_format=".mp4"):
    """
    Create an animation from the given files.

    If no file names are given, a file selection dialog will appear.

    Args:
        input_file_mask (basestring): file name mask with c-style format string
        input_file_names (iterable): names of the files

    Return:
        animation file
    """
    # TODO process user input on frame rate file format and so on
    if input_file_mask != "":
        output_name = "_".join(input_file_mask.split("_")[:-2]) + target_format
        args = ["-i", input_file_mask, "-c:v", "libx264", "-pix_fmt", "yuv420p", output_name]
        call(["ffmpeg"] + args)

        # ffmpeg -i Fri_Jun_24_16:14:50_2016_%04d.png transport_system.gif
        # convert Fri_Jun_24_16:14:50_2016_00*.png out.gif
