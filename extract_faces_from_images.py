import argparse
import os
import time

from PIL import Image

from detector import *
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from a video")
    parser.add_argument("--input_root", type=str)
    parser.add_argument("--result_root", type=str)
    parser.add_argument("--pad", type=float, default=1.0)
    args = parser.parse_args()

    make_folder(args.result_root)
    
    img_filenames = [os.path.join(args.input_root, i) \
                     for i in os.listdir(args.input_root)]

    for fn in img_filenames:
        img = Image.open(fn)
        cropped_faces = crop_faces(img, pad=args.pad)
        
        for idx, f in enumerate(cropped_faces, 1):
            save_filename = os.path.splitext(os.path.basename(fn))[0]
            if idx != 1:
                save_filename += f"-{idx}"
            f.save(os.path.join(args.result_root, save_filename + ".png"))