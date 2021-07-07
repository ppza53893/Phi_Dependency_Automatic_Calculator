"""
load a file, for VEE pro.
cerated by ohashi
IFileDialogでのフォルダ参照がC#だとめんどくさいので、pythonでやらせようという試み
"""

import os
import sys
from tkinter import Tk
from tkinter.filedialog import askdirectory


# pylint: disable=invalid-name
def get_directory(init_dir, svpath):
    """
    参照
    """
    root = Tk()
    root.withdraw()
    _path = askdirectory(initialdir=init_dir)
    if _path =='':
            exit(3)
    with open(svpath, 'w') as f:
            f.write(_path)


if __name__ == '__main__':
    fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.path')
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            init_dir = f.readline()
            if init_dir == '':
                init_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        init_dir = os.path.dirname(os.path.abspath(__file__))

    get_directory(init_dir, fpath)
