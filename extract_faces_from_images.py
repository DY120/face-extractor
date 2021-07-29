import argparse
import os
import time

import onnx
import onnxruntime as ort
from PIL import Image

from detector import *
from utils.misc import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract faces from a video")
    parser.add_argument("input_dir", type=str, default="samples")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--pad", type=float, default=1.0)
    parser.add_argument("--onnx_path", type=str, default="version-RFB-640.onnx")
    args = parser.parse_args()

    initialize_folder(args.result_dir)
    
    img_filenames = [os.path.join(args.input_dir, i) \
                     for i in os.listdir(args.input_dir)]

    model = onnx.load(args.onnx_path)
    onnx.checker.check_model(model)
    onnx.helper.printable_graph(model.graph)
    ort_session = ort.InferenceSession(args.onnx_path)

    for fn in img_filenames:
        img = Image.open(fn)
        #cropped_faces = rfb_crop_faces(img, ort_session=ort_session, pad=args.pad)
        cropped_faces = mtcnn_crop_faces(img, pad=args.pad)
        
        for idx, f in enumerate(cropped_faces, 1):
            save_filename = os.path.splitext(os.path.basename(fn))[0]
            if idx != 1:
                save_filename += f"-{idx}"
            f.save(os.path.join(args.result_dir, save_filename + ".png"))