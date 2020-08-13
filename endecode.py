import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import atexit
from os import path
import gym
from torch.utils.tensorboard import SummaryWriter

from models import StaticReconstructor, DiscriminatorConv
from utils import WarpFrame, NoopResetEnv, MaxAndSkipEnv

BATCH_SIZE = 1
SEQ_LEN = 500
NUM_STEPS = 10000 if torch.cuda.is_available() else 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'weights/endecode.pt'
GLR = 3e-4
DLR = 3e-4
L1_SCALER = 2.0
WEIGHT_DECAY = 0.0
WRITER = SummaryWriter(log_dir="logs/endecode-L1scaler"+str(L1_SCALER)+"-dLR"+str(DLR))

DATA = torch.zeros(NUM_STEPS, 1, 84, 84)


G = StaticReconstructor(lr=GLR, weight_decay=WEIGHT_DECAY, device=DEVICE)
G.to(DEVICE)
if path.exists(PATH):
    print("Loading model from", PATH)
    G.load_state_dict(torch.load(PATH, map_location=DEVICE))

D = DiscriminatorConv(lr=DLR, weight_decay=WEIGHT_DECAY, device=DEVICE)
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
    for k in range(5):
        # Generate batch of images for discriminator
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
    gout = G(z)
    G_gan_loss = torch.mean(torch.log(1 - D(gout)))
    G_l1_loss = L1_SCALER * torch.mean(torch.abs(gout - z))
    (G_gan_loss + G_l1_loss).backward()
    G.optim.step()
    # Log results
    WRITER.add_scalar('Accuracy/D Accuracy', np.mean(D.accuracy(x, G(z))), e)
    WRITER.add_scalar('Discriminator Loss/Loss Real', np.mean(D_loss_real.item()), e)
    WRITER.add_scalar('Discriminator Loss/Loss Fake', np.mean(D_loss_fake.item()), e)
    WRITER.add_scalar('Generator Loss/GAN Loss', np.mean(G_gan_loss.item()), e)
    WRITER.add_scalar('Generator Loss/L1 Loss', np.mean(G_l1_loss.item()), e)

