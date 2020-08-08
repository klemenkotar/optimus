import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import atexit
from os import path
import gym
from torch.utils.tensorboard import SummaryWriter

from models import StaticReconstructor, Descriminator
from utils import WarpFrame, NoopResetEnv, MaxAndSkipEnv

BATCH_SIZE = 1
SEQ_LEN = 1000
NUM_STEPS = 100000 if torch.cuda.is_available() else 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'weights/endecode.pt'
LR = 3e-4
WEIGHT_DECAY = 0.0
WRITER = SummaryWriter(log_dir="logs/endecode-no-z-high-lr")

DATA = torch.zeros(NUM_STEPS, 1, 84, 84)


def make_batch(start, n):
    tgt = DATA[idx:idx+n].to(DEVICE)
    return tgt, tgt


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


G = StaticReconstructor(lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE)
G.to(DEVICE)
if path.exists(PATH):
    print("Loading model from", PATH)
    G.load_state_dict(torch.load(PATH, map_location=DEVICE))

D = Descriminator(lr=1e-3, weight_decay=WEIGHT_DECAY, device=DEVICE)
D.to(DEVICE)


def exit_handler():
    print("Saving model as", PATH)
    torch.save(G.state_dict(), PATH)
    print("Saving Images")
    x = z = DATA[13]
    out = G(x.unsqueeze(0))
    tgt = z
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

print("Generating Data")
for step in tqdm(range(NUM_STEPS)):
    # Roll out env
    action = random.randint(0, 5)
    obs, rew, done, _ = env.step(action)
    DATA[step] = torch.tensor(obs.reshape(1, 84, 84))
    if done:
        env.reset()
DATA = DATA.to(DEVICE)

print("Training")
for e in tqdm(range(10000)):
    d_losses = []
    g_losses = []
    print("Epoch", e)

    # Generate batch of images
    x = DATA[torch.randperm(NUM_STEPS)[:SEQ_LEN]] / 255.0
    # z = DATA[torch.randperm(NUM_STEPS)[:SEQ_LEN]] / 255.0
    # Compute discriminator loss
    D.zero_grad()
    D_loss = -torch.mean(torch.log(D(x)) + torch.log(1 - D(G(x))))
    D_loss.backward()
    D.optim.step()
    # Generate batch of images for discriminator
    # z = DATA[torch.randperm(NUM_STEPS)[:SEQ_LEN]] / 255.0
    # Compute generator loss
    G.zero_grad()
    G_loss = -torch.mean(torch.log(1 - D(G(x))))
    G_loss.backward()
    G.optim.step()
    # Record losses
    d_losses.append((D_loss.item()))
    g_losses.append((G_loss.item()))
    WRITER.add_scalar('Loss/D Loss', np.mean(d_losses), e)
    WRITER.add_scalar('Loss/G Loss', np.mean(g_losses), e)
    print("D Loss: %.5f | G Loss: %.5f" % (np.mean(d_losses), np.mean(g_losses)))

# x = z = DATA[13]
# out = G(x.unsqueeze(0))
# tgt = z
# plt.figure(1)
# plt.imshow(tgt.squeeze().cpu().detach().numpy())
# plt.figure(2)
# plt.imshow(out.squeeze().cpu().detach().numpy())
# plt.show()
