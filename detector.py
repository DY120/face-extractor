import os
import shutil
import time

from PIL import Image

from mtcnn import detect_faces

def crop_faces(img, pad=1.0, conf=0.99):
    bounding_boxes, landmarks = detect_faces(img)
    cropped_faces = []

    for bb in bounding_boxes:
        confidence = bb[-1]
        if confidence < conf:
            continue

        width = (bb[2]-bb[0]) * pad
        height = (bb[3]-bb[1]) * pad

        center = ((bb[0]+bb[2]) / 2, (bb[1]+bb[3]) / 2)
        d = max(width, height)

        bb_ = [center[0] - d/2, center[1] - d/2, center[0] + d/2, center[1] + d/2]
        bb_ = [int(i) for i in bb_]

        cropped_faces.append(img.crop(bb_))

    return cropped_faces