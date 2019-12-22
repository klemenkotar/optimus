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

BATCH_SIZE = 16
SEQ_LEN = 100
NUM_STEPS = 100000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'models/rec.pt'
LR = 1e-3
WEIGHT_DECAY = 0.01

DATA = torch.zeros(NUM_STEPS, 84, 84)
ACTIONS = torch.zeros(NUM_STEPS, 1).long()

class Reconstruction(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 256, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3)),
        )
        self.action_encoder = nn.Embedding(32, 128)
        self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=12, dropout=0.1)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(640, 256, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=5),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 256, (1, 1))
        )


    def forward(self, x, act):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze()
        act = self.action_encoder(act)
        seq = torch.zeros((x.shape[0]*5, 128))
        for i in range(x.shape[0]):
            idx = i * 5
            seq[idx] = x[i, :128]
            seq[idx+1] = x[i, 128:256]
            seq[idx+2] = x[i, 256:384]
            seq[idx+3] = x[i, 384:512]
            seq[idx+4] = act[i]
        seq = seq.unsqueeze(1)
        out = self.transformer(seq, seq)
        out = out.squeeze()
        out = out.view(x.shape[0], 128*5, 1, 1)
        out = self.deconv(out)
        return out

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac, gaze=None):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        # gym.Wrapper.__init__(self, env)
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action, gaze=None):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            if gaze is None:
                obs, reward, done, info = self.env.step(action)
            else:
                obs, reward, done, info = self.env.step(action, gaze=gaze)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def make_batch(start, n):
    tgt = DATA[idx:idx+n].to(DEVICE)
    act = ACTIONS[idx:idx+n-1].to(DEVICE)
    return tgt[:-1], tgt[1:], act

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

model = Reconstruction()
model.to(DEVICE)
if path.exists(PATH):
    print("Loading model from", PATH)
    model.load_state_dict(torch.load(PATH))

optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def exit_handler():
    print("Saving model as", PATH)
    torch.save(model.state_dict(), PATH)
# atexit.register(exit_handler)

env = gym.make("PongNoFrameskip-v4")
env = WarpFrame(env, width=84, height=84)
env = NoopResetEnv(env)
env = MaxAndSkipEnv(env)
env.reset()
step = 0
while step < NUM_STEPS:
    # Roll out env
    for i in range(SEQ_LEN * 16):
        action = random.randint(0, 5)
        obs, rew, done, _ = env.step(action)
        DATA[step] = torch.tensor(obs.squeeze())
        ACTIONS[step] = action
        step += 1
        if done:
            env.reset()
    train_losses = []
    # train
    print("Training")
    for idx in tqdm(generate_batch_indexes(0, step, SEQ_LEN)):
        optim.zero_grad()
        seq, tgt, act = make_batch(idx, SEQ_LEN)
        out = model(seq, act)
        out = out.permute(0, 2, 3, 1).reshape(-1, 256)
        tgt = tgt.view(-1).long()
        loss = F.cross_entropy(out, tgt)
        loss.backward()     
        optim.step()
        train_losses.append(loss.item())
    print("Loss:", np.mean(train_losses))

seq, tgt, act = make_batch(idx, SEQ_LEN)
out = model(seq, act)
tgt = tgt[0]
out = torch.argmax(out[0].permute(2,3,1), dim=2)
plt.figure(1)
plt.imshow(tgt.squeeze().cpu().detach().numpy())
plt.savefig('tgt-model')
plt.figure(2)
plt.imshow(out.squeeze().cpu().detach().numpy())
plt.savefig('out-model')