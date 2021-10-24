import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import random
from mlxtend.data import loadlocal_mnist
from net import Generator, Discriminator


def load_data():
    train, _ = loadlocal_mnist('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    test, _ = loadlocal_mnist('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
    return (np.concatenate((train, test), axis=0).reshape((-1, 1, 28, 28)) - 127.5) / 127.5


def random_batch(data, batch_size):
    idx = np.random.choice(data.shape[0], batch_size, False)
    return data[idx]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 0.0002
k = 1
noise_size = 100
n_iter = 6000
batch_size = 128
distribution = torch.distributions.Uniform(-1, 1)

data = load_data()
g = Generator(noise_size).to(device)
d = Discriminator().to(device)
fixed_noise = distribution.sample((1, noise_size, 1, 1)).to(device)

optimg = optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
optimd = optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))

g.train()
d.train()

for step in range(n_iter):
    de = 0
    dn = 0
    for i in range(k):
        noise = distribution.sample((batch_size, noise_size, 1, 1)).to(device)
        examples = torch.FloatTensor(random_batch(data, batch_size)).to(device)
        disc_e = d(examples)
        disc_n = d(g(noise))
        de += disc_e.mean().item()
        dn += disc_n.mean().item()
        loss = - (torch.log(disc_e) + torch.log(1 - disc_n))
        loss = loss.mean()
        optimd.zero_grad()
        loss.backward()
        optimd.step()
    noise = distribution.sample((batch_size, noise_size, 1, 1)).to(device)
    loss = torch.log(1 - d(g(noise)))
    loss = loss.mean()
    optimg.zero_grad()
    loss.backward()
    optimg.step()
    print('Step %d | Disc on examples: %.3f | Disc on generated: %.3f' % (step, de / k, dn / k))
    if step % 25 == 0:
        g.eval()
        img = g(fixed_noise)
        img = img.detach().cpu().numpy()[0, 0]
        cv2.imwrite('fixed.png', (127.5 * img) + 127.5)
        g.train()
        torch.save(g, 'gen.pkl')
        torch.save(d, 'disc.pkl')