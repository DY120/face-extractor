import argparse

import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from backbones import get_model


@torch.no_grad()
def inference(img, net):
    img = ToTensor()(img).unsqueeze(0).float()
    img.sub_(0.5).div_(0.5)

    output = net(img)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='backbone.pth')
    args = parser.parse_args()

    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weight))
    net.eval()

    img = Image.open(args.img).resize((112, 112))
    inference(img, net)