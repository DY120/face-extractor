import argparse
import os
import time

import cv2
import onnx
import onnxruntime as ort
from PIL import Image

from detector import *
from utils.misc import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from a video")
    parser.add_argument("input_path", type=str)
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--pad", type=float, default=1.0)
    parser.add_argument("--min_size", type=int, default=0)
    parser.add_argument("--num_sample", type=int, default=-1)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--onnx_path", type=str, default="version-RFB-640.onnx")
    args = parser.parse_args()

    initialize_folder(args.result_dir)

    model = onnx.load(args.onnx_path)
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)
    ort_session = ort.InferenceSession(args.onnx_path)

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
                cropped_faces = rfb_crop_faces(frame, ort_session=ort_session, pad=args.pad)
                #cropped_faces = mtcnn_crop_faces(img, pad=args.pad)
            except:
                continue

            for idx, f in enumerate(cropped_faces, 1):
                if min(f.size) < args.min_size:
                    continue

                save_filename = str(face_count)
                if idx != 1:
                    save_filename += f"-{idx}"
                f.save(os.path.join(args.result_dir, save_filename + ".png"))
                face_count += 1

            print(f"{frame_count:06d}/{num_frame:06d}")
        
        frame_count += 1

        if frame_count == args.num_sample:
            break