""" Models """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Reconstructor(nn.Module):

    def __init__(self, lr=1e-4, weight_decay=0.0, device=torch.device("cpu")):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (2, 2), stride=2),
            nn.ReLU()
        )

        self.action_encoder = nn.Embedding(32, 128)
        encoder_layer = torch.nn.TransformerEncoderLayer(128, 8)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, 8)
        self.transformer = nn.Transformer(d_model=128, nhead=8, dropout=0.2)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 256, (1, 1))
        )

        x_grid = np.reshape(np.arange(-1, 1.0001, 2/83), (1, 84))
        x_grid = torch.tensor(np.repeat(x_grid, 84, axis=0))
        y_grid = torch.rot90(x_grid, -1)
        self.grid = torch.stack((torch.zeros_like(x_grid), x_grid, y_grid), axis=0)

        self.big_to_smol = nn.Linear(128, 64)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.device=device

    def forward(self, x, act):

        # Add grid to input
        # grid = self.grid.repeat(x.shape[0], 1, 1, 1).float().to(DEVICE)
        # grid[:, 0, :, :] = x

        # Pass inputs through conv
        x = x.unsqueeze(1)
        x = self.conv(x)

        # Convert actions into action embeddings
        act = self.action_encoder(act)

        # Construct transformer sequence from conv outputs
        seq = torch.zeros((x.shape[0]*5, 128)).to(self.device)
        for i in range(x.shape[0]):
            idx = i * 5
            seq[idx] = x[i, :, 0, 0]
            seq[idx+1] = x[i, :, 0, 1]
            seq[idx+2] = x[i, :, 1, 0]
            seq[idx+3] = x[i, :, 1, 1]
            seq[idx+4] = act[i]

        ### We have the linear embedding sequence here
        # plt.imshow(seq.detach().cpu().numpy().swapaxes(0,1))
        # plt.show()
        # exit()
        seq = seq.unsqueeze(1)
        trans_out = self.encoder(seq)
        seq = seq.squeeze()
        trans_out = seq

        # Construct conv inputs for reconstruction
        deconv_in = torch.zeros((x.shape[0], 128, 2, 2)).to(self.device)
        for i in range(x.shape[0]):
            idx = (i * 5)
            deconv_in[i, :, 0, 0] = trans_out[idx] * trans_out[idx+4]
            deconv_in[i, :, 0, 1] = trans_out[idx+1] * trans_out[idx+4]
            deconv_in[i, :, 1, 0] = trans_out[idx+2] * trans_out[idx+4]
            deconv_in[i, :, 1, 1] = trans_out[idx+3] * trans_out[idx+4]

        # Deconvolve embeddings
        out = self.deconv(deconv_in)

        return out


class StaticReconstructor(nn.Module):

    def __init__(self, lr=1e-4, weight_decay=0.0, device=torch.device("cpu")):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (2, 2), stride=2),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 256, (1, 1))
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device

    def forward(self, x):
        return self.deconv(self.conv(x))


class Descriminator(nn.Module):

    def __init__(self, lr=3e-6, weight_decay=0.0, device=torch.device("cpu")):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(84*84, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.optim = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=0.0)
        self.device = device

    def forward(self, x):
        return self.discriminator(x)
