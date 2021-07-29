import os
import shutil
import time

import numpy as np
from PIL import Image

from mtcnn import detect_faces
import utils.box_utils as box_utils

def mtcnn_crop_faces(img, pad=1.0, conf=0.99):
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

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue

        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])

    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def rfb_crop_faces(img, ort_session=None, pad=1.0, conf=0.99, threshold=0.7):
    """
    Input: A PIL image \\
    Output: A list of PIL face images cropped from the input image
    """

    img_w, img_h = img.size
    dx, dy = 0, 0

    if 3 * img_w > 4 * img_h:
        dy = img_w * 3 // 4 - img_h
    else:
        dx = 4 * img_h // 3 - img_w

    img_padded = Image.new(img.mode, (img_w + dx, img_h + dy), (0, 0, 0))
    img_padded.paste(img, (0, 0))
    img_resize = img_padded.resize((640, 480))
    scale_factor = img_padded.size[0] / img_resize.size[0]
    scale_factor = 1
    
    np_img_resize = np.asarray(img_resize)
    img_mean = np.array([127, 127, 127])
    np_img_resize = (np_img_resize - img_mean) / 128
    np_img_resize = np.transpose(np_img_resize, [2, 0, 1])
    np_img_resize = np.expand_dims(np_img_resize, axis=0)
    np_img_resize = np_img_resize.astype(np.float32)
    
    input_name = ort_session.get_inputs()[0].name
    confidences, boxes = ort_session.run(None, {input_name: np_img_resize})

    bounding_boxes, labels, probs = \
        predict(img_padded.size[0], img_padded.size[1], confidences, boxes, threshold)
    cropped_faces = []

    for bb, prob in zip(bounding_boxes, probs):
        confidence = prob
        if confidence < conf:
            continue

        width = (bb[2]-bb[0]) * pad * scale_factor
        height = (bb[3]-bb[1]) * pad * scale_factor

        center = ((bb[0]+bb[2]) * (scale_factor/2), 
                  (bb[1]+bb[3]) * (scale_factor/2))
        d = max(width, height)

        bb_ = [center[0] - d/2, center[1] - d/2, center[0] + d/2, center[1] + d/2]
        bb_ = [int(i) for i in bb_]

        cropped_faces.append(img_padded.crop(bb_))

    return cropped_faces