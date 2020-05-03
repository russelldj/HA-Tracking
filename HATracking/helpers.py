import os
import pdb


def unique_output_filename(folder, format_string, maxval=100000):
    """
    writes out data which we don't want to be overwritten
    folder : str
        The folder to write to
    format_string : str
        Something which can be formated using format_string.format(i)
    maxval : int
        The maximum number of iterations to try
    """
    if not os.path.isdir(folder):
        raise ValueError("Folder {} does not exists".format(folder))
    full_format_string = os.path.join(folder, format_string)
    for i in range(maxval):
        if not os.path.exists(full_format_string.format(i)):
            return full_format_string.format(i)

    raise ValueError("No file could be created with folder {}, format string {}, and maxval {}".format(
        folder, format_string, maxval))
