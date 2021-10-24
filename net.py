import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.noise_size = noise_size
        self.conv1 = nn.ConvTranspose2d(noise_size, 512, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))
        self.b1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.b2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
        self.b3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.b4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        # b x noise_size x 1 x 1
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.b2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.b3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.b4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.b1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
        self.b2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.b3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0))

    def forward(self, x):
        # b x 1 x 28 x 28
        x = self.conv1(x)
        x = self.b1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = self.b2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = self.b3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 1)

        return x
