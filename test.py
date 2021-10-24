import torch
import torch.nn as nn
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mlxtend.data import loadlocal_mnist
from net import Generator, Discriminator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
noise_size = 100
distribution = torch.distributions.Uniform(-1, 1)
g = torch.load('gen.pkl').to(device)
g.eval()

x = 10
y = 10
noise = distribution.sample((x * y, noise_size, 1, 1)).to(device)
img = g(noise).detach().cpu().numpy()

f, ax = plt.subplots(x, y)
for i in range(x):
    for j in range(y):
        ax[i, j].imshow(img[i * y + j, 0], cmap='gray')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])


plt.savefig('generated.png')
