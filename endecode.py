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
NUM_STEPS = 10000 if torch.cuda.is_available() else 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'weights/endecode.pt'
LR = 3e-4
WEIGHT_DECAY = 0.0
WRITER = SummaryWriter(log_dir="logs/endecode")

DATA = torch.zeros(NUM_STEPS, 1, 84, 84)


G = StaticReconstructor(lr=LR, weight_decay=WEIGHT_DECAY, device=DEVICE)
G.to(DEVICE)
if path.exists(PATH):
    print("Loading model from", PATH)
    G.load_state_dict(torch.load(PATH, map_location=DEVICE))

D = Descriminator(lr=3e-2, weight_decay=WEIGHT_DECAY, device=DEVICE)
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
    # Generate batch of images
    x = DATA[torch.randperm(NUM_STEPS)[:SEQ_LEN]] / 255.0
    z = DATA[torch.randperm(NUM_STEPS)[:SEQ_LEN]] / 255.0
    # Compute discriminator loss
    D.optim.zero_grad()
    D_loss_real = -torch.mean(torch.log(D(x)))
    D_loss_fake = -torch.mean(torch.log(1 - D(G(z))))
    (D_loss_real + D_loss_fake).backward()
    D.optim.step()
    # Generate batch of images for generator
    z = DATA[torch.randperm(NUM_STEPS)[:SEQ_LEN]] / 255.0
    # Compute generator loss
    G.optim.zero_grad()
    G_loss = torch.mean(torch.log(1 - D(G(z))))
    G_loss.backward()
    G.optim.step()
    # Log results
    WRITER.add_scalar('Accuracy/D Accuracy', np.mean(D.accuracy(x, G(z))), e)
    WRITER.add_scalar('Loss/D Loss Real', np.mean(D_loss_real.item()), e)
    WRITER.add_scalar('Loss/D Loss Fake', np.mean(D_loss_fake.item()), e)
    WRITER.add_scalar('Loss/G Loss', np.mean(G_loss.item()), e)

# x = z = DATA[13]
# out = G(x.unsqueeze(0))
# tgt = z
# plt.figure(1)
# plt.imshow(tgt.squeeze().cpu().detach().numpy())
# plt.figure(2)
# plt.imshow(out.squeeze().cpu().detach().numpy())
# plt.show()
