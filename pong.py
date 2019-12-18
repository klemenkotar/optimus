import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np

FILE = np.load('data/embeddings.npy', mmap_mode='r')
BATCH_SIZE = 1
SEQ_LEN = 1000
NUM_EPOCHS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_batch(idx, n, batch_size=1):
    seq = torch.tensor(FILE[idx:idx+(n*batch_size)]).to(DEVICE)
    seq = seq.view(n, batch_size, -1).float()
    return seq[:n//2], seq

def generate_batch_indexes(start, stop, step):
    idxs = []
    idx = start
    while idx < stop:
        idxs.append(idx)
        idx += step
    random.shuffle(idxs)
    return idxs

transfomer = nn.Transformer(d_model=528, nhead=16, num_encoder_layers=12).to(DEVICE)
optim = torch.optim.Adam(transfomer.parameters())

for e in range(NUM_EPOCHS):
    train_losses = []
    test_losses = []
    # train
    for idx in generate_batch_indexes(0, 900000, SEQ_LEN * BATCH_SIZE):
        optim.zero_grad()
        seq, tgt = make_batch(idx, SEQ_LEN, batch_size=BATCH_SIZE)
        out = transfomer(seq, tgt)
        # compute the 3 different loss functions
        emb_loss = F.binary_cross_entropy(torch.sigmoid(out[:,:,:512]), torch.sigmoid(tgt[:,:,:512]))
        print("idx:", idx, " --- out shape:", out[:,:,512:518].squeeze().shape, " --- tgt shape:", torch.argmax(tgt[:,:,512:518].squeeze(), dim=1).shape)
        action_loss = F.cross_entropy(out[:,:,512:518].squeeze(), torch.argmax(tgt[:,:,512:518].squeeze(), dim=1))
        value_loss = F.mse_loss(out[:,:,518], tgt[:,:,518])
        loss = emb_loss + action_loss + value_loss
        loss.backward()     
        optim.step()
        train_losses.append(loss.item())
        print(loss.item())
    for idx in generate_batch_indexes(900000, 1000000, SEQ_LEN * BATCH_SIZE):
        seq, tgt = make_batch(idx, SEQ_LEN, batch_size=BATCH_SIZE)
        out = transfomer(seq, tgt)
        # compute the 3 different loss functions
        emb_loss = F.binary_cross_entropy(torch.sigmoid(out[:,:,:512]), torch.sigmoid(tgt[:,:,:512]))
        action_loss = F.cross_entropy(out[:,:,512:518], torch.argmax(tgt[:,:,512:518]))
        value_loss = F.mse_loss(out[:,:,518], tgt[:,:,518])
        loss = emb_loss + action_loss + value_loss
        loss.backward()
        test_losses.append(loss.item())
    print("Epoch:", e+1, "\tTrain Loss:", np.mean(train_losses), "\tTest Loss:", np.mean(test_losses))

seq, tgt = make_batch(90000, SEQ_LEN, batch_size=1)
out = transfomer(seq, tgt)
plt.figure(1)
plt.imshow(tgt.squeeze().cpu().detach().numpy().swapaxes(0,1))
plt.savefig('tgt')
plt.figure(2)
plt.imshow(out.squeeze().cpu().detach().numpy().swapaxes(0,1))
plt.savefig('out')
