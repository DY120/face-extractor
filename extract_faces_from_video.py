import argparse
import os
import time

import cv2
from PIL import Image

from detector import *
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from a video")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--result_root", type=str)
    parser.add_argument("--pad", type=float, default=1.0)
    parser.add_argument("--min_size", type=int, default=0)
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    make_folder(args.result_root)

    cap = cv2.VideoCapture(args.input_path)
    frame_count = 0
    face_count = 0
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        retval, frame = cap.read()
        if not retval: 
            break

        if frame_count % args.interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            
            try:
                cropped_faces = crop_faces(frame, pad=args.pad)
            except:
                continue

            for idx, f in enumerate(cropped_faces, 1):
                if min(f.size) < args.min_size:
                    continue

                save_filename = str(face_count)
                if idx != 1:
                    save_filename += f"-{idx}"
                f.save(os.path.join(args.result_root, save_filename + ".png"))
                face_count += 1

            print(f"{frame_count:06d}/{num_frame:06d}")
        
        frame_count += 1

        if frame_count == args.num_sample:
            break