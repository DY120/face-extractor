import os
import shutil

def initialize_folder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    
    os.mkdir(folder)

def get_filenames(input_dir):
    img_filenames = [os.path.join(input_dir, i) \
                     for i in os.listdir(input_dir)]

    return img_filenames