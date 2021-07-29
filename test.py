import os

from PIL import Image
import torch
import torch.nn as nn

from backbones import get_model
from recognition import inference

network = "r50"
weight = "backbone.pth"

net = get_model(network, fp16=False)
net.load_state_dict(torch.load(weight))
net.eval()

cos = nn.CosineSimilarity()

img_filenames = [os.path.join("results_8k_", i) \
                 for i in os.listdir("results_8k_")]

references = []

for i in range(3):
    i = str(i) if i > 1 else ''
    img = Image.open(f"results/iu{i}.png").resize((112, 112))
    out = inference(img, net)
    references.append(out)

for fn in img_filenames:
    img2 = Image.open(fn).resize((112, 112))
    out2 = inference(img2, net)

    sum = 0
    for rf in references:
        sum += (cos(rf, out2).item() / len(references))

    print(fn, sum)