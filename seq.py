import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np

NUMBITS = 32

def int_to_emb(x, n):
    emb = torch.zeros(n).float()
    binint = bin(x)
    embidx = n-1
    for stridx in reversed(range(2, len(binint))):
        emb[embidx] = 0 if binint[stridx] == '0' else 1
        embidx -= 1
    return emb

def gen_seq(start, n):
    embeddings = []
    for i in range(n):
        embeddings.append(int_to_emb(start+i, NUMBITS))
    return torch.stack(embeddings, dim=0)

def make_batch(n, batch_size=32):
    seqs = []
    for _ in range(batch_size):
        start = random.randint(0, 10000000)
        seq = gen_seq(start, n)
        seqs.append(seq)
    seqs = torch.stack(seqs, dim=1)
    return seqs[:n//2], seqs

# transfomer = nn.Transformer(d_model=NUMBITS, nhead=16, num_encoder_layers=12)
encoder_layers = nn.TransformerEncoderLayer(32, 8)
model = nn.TransformerEncoder(encoder_layers, 8)
optim = torch.optim.Adam(model.parameters())

for i in range(100):
    optim.zero_grad()
    seq, tgt = make_batch(256)
    out = model(seq)
    loss = F.mse_loss(out, tgt)
    loss.backward()
    optim.step()
    print("Epoch:", i+1, "\tLoss:", loss.item())

seq, tgt = make_batch(256, batch_size=1)
out = model(seq)
plt.figure(1)
plt.imshow(tgt.squeeze().detach().numpy().swapaxes(0,1))
plt.figure(2)
out = F.sigmoid(out)
out = out.round()
plt.imshow(out.squeeze().detach().numpy().swapaxes(0,1))
plt.show()
