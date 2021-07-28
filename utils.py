import os
import shutil

def make_folder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    
    os.mkdir(folder)