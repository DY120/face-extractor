import os
import shutil

def initialize_folder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    
    os.mkdir(folder)