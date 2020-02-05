import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import atexit
from os import path
import gym
import cv2

from models import Reconstructor, Descriminator
from utils import WarpFrame, NoopResetEnv, MaxAndSkipEnv

BATCH_SIZE = 1
SEQ_LEN = 100
NUM_STEPS = 20000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'weights/atari.pt'
LR = 1e-4
WEIGHT_DECAY = 0.0

DATA = torch.zeros(NUM_STEPS, 84, 84)
ACTIONS = torch.zeros(NUM_STEPS, 1).long()


def make_batch(start, n):
    tgt = DATA[idx:idx+n].to(DEVICE)
    act = ACTIONS[idx:idx+n].to(DEVICE)
    return tgt, tgt, act

def make_embedding_batch(idx, n, batch_size=1):
    tgt = DATA[idx:idx+n].to(DEVICE)
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

model = Reconstruction(lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE)
model.to(DEVICE)
if path.exists(PATH):
    print("Loading model from", PATH)
    model.load_state_dict(torch.load(PATH, map_location=DEVICE))

D = Descriminator(lr=3e-6, weight_decay=WEIGHT_DECAY, device=DEVICE)
D.to(DEVICE)

def exit_handler():
    print("Saving model as", PATH)
    torch.save(model.state_dict(), PATH)
    print("Saving Images")
    seq, tgt, act = make_batch(idx, SEQ_LEN)
    out = model(seq, act)
    tgt = tgt[0]
    out = torch.argmax(out[0].permute(1,2,0), dim=2)
    plt.figure(1)
    plt.imshow(tgt.squeeze().cpu().detach().numpy())
    plt.savefig('tgt-atari')
    plt.figure(2)
    plt.imshow(out.squeeze().cpu().detach().numpy())
    plt.savefig('out-atari')
atexit.register(exit_handler)

env = gym.make("PongNoFrameskip-v4")
env = WarpFrame(env, width=84, height=84)
env = NoopResetEnv(env)
env = MaxAndSkipEnv(env)
env.reset()
step = 0
while step < NUM_STEPS:
    # Roll out env
    for i in range(SEQ_LEN * 10):
        action = random.randint(0, 5)
        obs, rew, done, _ = env.step(action)
        DATA[step] = torch.tensor(obs.squeeze())
        ACTIONS[step] = action
        step += 1
        if done:
            env.reset()

for e in range(300):
    train_losses = []
    rec_losses = []
    d_losses = []
    g_losses = []
    print("Epoch", e)

    for idx in tqdm(generate_batch_indexes(0, len(DATA), SEQ_LEN)):
        seq, tgt, act = make_batch(idx, SEQ_LEN)
        out = model(seq, act)
        gt_out = D(seq)
        rec_out = D(torch.argmax(out, dim=1).unsqueeze(1).float())
        out = out.permute(0, 2, 3, 1).reshape(-1, 256)
        tgt = tgt.view(-1).long()

        # Update generator
        model.optim.zero_grad()
        rec_loss = F.cross_entropy(out, tgt)
        g_loss = -(torch.log(1 - rec_out)).mean()
        generator_loss = rec_loss + g_loss
        generator_loss.backward(retain_graph=True)
        model.optim.step()

        # Update discriminator
        D.optim.zero_grad()
        d_loss = -(torch.log(gt_out) + torch.log(1.0 - rec_out)).mean()
        discriminator_loss = d_loss
        discriminator_loss.backward()
        D.optim.step()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        rec_losses.append(rec_loss.item())
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        train_losses.append(rec_loss.item() + d_loss.item() + g_loss.item())

    print("Loss: %.5f | Rec Loss: %.5f | D Loss: %.5f | G Loss: %.5f" % 
        (np.mean(train_losses), np.mean(rec_losses), np.mean(d_losses), np.mean(g_losses)))
