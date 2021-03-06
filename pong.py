import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import atexit
from os import path

FILE = np.load('data/emb5.npy')
BATCH_SIZE = 50
SEQ_LEN = 100
NUM_EPOCHS = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'model5.pt'
LR = 1e-4
WEIGHT_DECAY = 0.01

def make_batch(idx, n, batch_size=1):
    tgt = torch.tensor(FILE[idx:idx+(n*batch_size)]).to(DEVICE)
    tgt = tgt.view(n, batch_size, -1).float()
    # tgt = torch.clamp(torch.round(tgt), 0.0, 1.0)
    seq = tgt.detach().clone()
    seq[(torch.randint(0, n//2 -1 , (n//8,)) * 2) + 1] = 0.0 #-float("inf")
    return seq[:-1], tgt[:-1], tgt[1:]

def generate_batch_indexes(start, stop, step):
    idxs = []
    idx = start
    while idx < stop:
        tidx = idx + random.randint(-1000, 1000)
        tidx = max(0, tidx)
        tidx = min(stop-step, tidx)
        idxs.append(tidx)
        idx += step
    random.shuffle(idxs)
    return idxs

transfomer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=12, dropout=0.5).to(DEVICE)
if path.exists(PATH):
    print("Loading model from", PATH)
    transfomer.load_state_dict(torch.load(PATH))

optim = torch.optim.Adam(transfomer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def exit_handler():
    print("Saving model as", PATH)
    torch.save(transfomer.state_dict(), PATH)
atexit.register(exit_handler)

for e in range(NUM_EPOCHS):
    train_losses = []
    test_losses = []
    test_action_loss = []
    test_value_loss = []
    test_emb_loss = []
    # train
    print("Training")
    for idx in tqdm(generate_batch_indexes(0, 180000, SEQ_LEN * BATCH_SIZE)):
        optim.zero_grad()
        seq, gt, tgt = make_batch(idx, SEQ_LEN, batch_size=BATCH_SIZE)
        out = transfomer(seq, gt)
        # compute the 3 different loss functions
        # emb_loss = F.l1_loss(out[:,:,:512], tgt[:,:,:512])
        # loss = F.cross_entropy(out[:,:,512:518].view(out.shape[0] * out.shape[1], -1), torch.argmax(tgt[:,:,512:518].view(out.shape[0] * out.shape[1], -1), dim=1))
        # value_loss = F.mse_loss(out[:,:,518], tgt[:,:,518])
        # loss = emb_loss + action_loss + value_loss
        loss = F.l1_loss(out[torch.arange((SEQ_LEN-1)//2) * 2, :], tgt[torch.arange((SEQ_LEN-1)//2) * 2, :])
        loss.backward()     
        optim.step()    
        train_losses.append(loss.item())
    print("Testing")
    for idx in tqdm(generate_batch_indexes(180000, 200000, SEQ_LEN * BATCH_SIZE)):
        seq, gt, tgt = make_batch(idx, SEQ_LEN, batch_size=BATCH_SIZE)
        out = transfomer(seq, gt)
        # compute the 3 different loss functions
        # emb_loss = F.l1_loss(out[:,:,:512], tgt[:,:,:512])
        # loss = F.cross_entropy(out[:,:,512:518].view(out.shape[0] * out.shape[1], -1), torch.argmax(tgt[:,:,512:518].view(out.shape[0] * out.shape[1], -1), dim=1))
        # value_loss = F.mse_loss(out[:,:,518], tgt[:,:,518])
        # test_emb_loss.append(emb_loss.item())
        # test_action_loss.append(action_loss.item())
        # test_value_loss.append(value_loss.item())
        # loss = emb_loss + action_loss + value_loss
        loss = F.l1_loss(out[torch.arange((SEQ_LEN-1)//2) * 2, :], tgt[torch.arange((SEQ_LEN-1)//2) * 2, :])
        test_losses.append(loss.item())
    print("Epoch:", e+1, "\tTrain Loss:", np.mean(train_losses), "\tTotal Test Loss:", np.mean(test_losses))
    # print("Emb Loss:", np.mean(test_emb_loss), "\tAction Loss:", np.mean(test_action_loss), "\tValue Loss:", np.mean(test_value_loss))

seq, gt, tgt = make_batch(0, SEQ_LEN, batch_size=1)
out = transfomer(seq, gt)
plt.figure(1)
plt.imshow(tgt.squeeze().cpu().detach().numpy().swapaxes(0,1))
plt.savefig('tgt5')
plt.figure(2)
plt.imshow(out.squeeze().cpu().detach().numpy().swapaxes(0,1))
plt.savefig('out5')
