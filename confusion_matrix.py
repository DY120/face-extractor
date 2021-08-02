import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import plot_confusion_matrix
import torch
import torch.nn as nn

from backbones.model_irse import IR_50
from recognition import inference
from utils.misc import *

img_filenames = get_filenames("results")
n_imgs = len(img_filenames)
print(img_filenames)

network = "r50"
weight = "backbone_ir50_asia.pth"
net = IR_50((112, 112))
net.load_state_dict(torch.load(weight))
net.eval()

cos = nn.CosineSimilarity()

features = [inference(Image.open(i).resize((112, 112)), net) \
            for i in img_filenames]
conf_matrix = np.zeros((n_imgs, n_imgs))

for i in range(n_imgs):
    for j in range(n_imgs):
        conf_matrix[i][j] = cos(features[i], features[j])

plt.matshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.show()