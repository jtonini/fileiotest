# -*- coding: utf-8 -*-
from   typing import *

import argparse
import base64
import tempfile

###
# Standard imports, starting with os and sys
###
min_py = (3, 9)
import os
import sys
if sys.version_info < min_py:
    print(f"This program requires Python {min_py[0]}.{min_py[1]}, or higher.")
    sys.exit(os.EX_SOFTWARE)

###
# Credits
###
__author__ = 'George Flanagin'
__copyright__ = 'Copyright 2024, University of Richmond'
__credits__ = None
__version__ = 0.1
__maintainer__ = 'George Flanagin'
__email__ = 'gflanagin@richmond.edu'
__status__ = 'in progress'
__license__ = 'MIT'

def random_string(length:int=10, want_bytes:bool=False, all_alpha:bool=True) -> str:
    """
    Generates a random string or byte sequence of the specified length.

    Parameters:
    length (int): The desired length of the output string or byte sequence. Default is 10.
    want_bytes (bool): If True, returns the result as a bytes object. If False, returns a string. Default is False.
    all_alpha (bool): If True, ensures the resulting string consists only of alphabetic characters (a-z, A-Z).
                      If False, the resulting string may include non-alphabetic characters. Default is True.

    Returns:
    str: A random string of the specified length if `want_bytes` is False.
    bytes: A random byte sequence of the specified length if `want_bytes` is True.

    """

    s = base64.b64encode(os.urandom(length*2))
    if want_bytes: return s[:length]

    s = s.decode('utf-8')
    if not all_alpha: return s[:length]

    t = "".join([ _ for _ in s if _.isalpha() ])[:length]
    return t


def random_file(name_prefix:str, *, length:int=None, break_on:str=None) -> tuple:
    """
    Generate a new file, with random contents, consisting of printable
    characters.

    name_prefix -- In case you want to isolate them later.
    length -- if None, then a random length <= 1MB
    break_on -- For some testing, perhaps you want a file of "lines."

    returns -- a tuple of file_name and size.
    """
    f_name = None
    num_written = -1

    file_size = length if length is not None else random.choice(range(0, 1<<20))
    s = random_string(file_size, True)

    if break_on is not None:
        if isinstance(break_on, str): break_on = break_on.encode('utf-8')
        s = s.replace(break_on, b'\n')

    try:
        f_no, f_name = tempfile.mkstemp(suffix='.iotest', prefix=name_prefix, dir=os.getcwd())
        num_written = os.write(f_no, s)
        os.close(f_no)
    except Exception as e:
        print(f"{e=}")

    return f_name, num_written


def randomfiles_main(num:int) -> int:

    files = {}
    for i in range(num):
        name, size = random_file('randfile', length=10000000)
        files[name] = size

    # print(files)

    return os.EX_OK


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, required=True,
        help="Number of random files required.")

    myargs = parser.parse_args()
    sys.exit(randomfiles_main(myargs.num))
