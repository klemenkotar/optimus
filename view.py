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
from PIL import Image
import glob

SEQ_LEN = 100
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'models/rec-res.pt'

DATA = torch.zeros(100, 84, 84, device=DEVICE)
ACTIONS = torch.zeros(100, 1, device=DEVICE, dtype=torch.long)

class Reconstruction(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (4, 4), stride=2, padding=2),
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
            nn.Conv2d(128, 128, (4, 4), stride=1),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(3, 64, (4, 4), stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 64, (4, 4), stride=1)
        self.conv3 = nn.Conv2d(64, 128, (4, 4), stride=2)
        self.conv4 = nn.Conv2d(128, 128, (4, 4), stride=1)
        self.conv5 = nn.Conv2d(128, 128, (4, 4), stride=1)
        self.conv6 = nn.Conv2d(128, 128, (4, 4), stride=1)
        self.conv7 = nn.Conv2d(128, 128, (4, 4), stride=2)
        self.conv8 = nn.Conv2d(128, 128, (2, 2), stride=2)

        self.action_encoder = nn.Embedding(32, 128)
        encoder_layer = torch.nn.TransformerEncoderLayer(128, 8)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, 8)
        self.transformer = nn.Transformer(d_model=128, dropout=0.2)
        self.deconv = nn.Sequential(
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

        self.relu = nn.ReLU()
        self.deconv1 = nn.ConvTranspose2d(128, 128, (2, 2), stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 128, (4, 4), stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 128, (4, 4), stride=1)
        self.deconv4 = nn.ConvTranspose2d(128, 128, (4, 4), stride=1)
        self.deconv5 = nn.ConvTranspose2d(128, 128, (4, 4), stride=1)
        self.deconv6 = nn.ConvTranspose2d(128, 64, (4, 4), stride=2)
        self.deconv7 = nn.ConvTranspose2d(64, 64, (4, 4), stride=1)
        self.deconv8 = nn.ConvTranspose2d(64, 64, (4, 4), stride=2, padding=2)
        self.deconv9 = nn.ConvTranspose2d(64, 256, (1, 1))

        x_grid = np.reshape(np.arange(-1, 1.0001, 2/83), (1, 84))
        x_grid = torch.tensor(np.repeat(x_grid, 84, axis=0))
        y_grid = torch.rot90(x_grid, -1)
        self.grid = torch.stack((torch.zeros_like(x_grid), x_grid, y_grid), axis=0)

        self.big_to_smol = nn.Linear(128, 64)


    def forward(self, x, act):

        # Add grid to input
        grid = self.grid.repeat(x.shape[0], 1, 1, 1).float().to(DEVICE)
        grid[:, 0, :, :] = x

        # Pass inputs through conv
        # x = x.unsqueeze(1)
        # x = self.conv(grid).squeeze()
        conv1_out = self.relu(self.conv1(grid))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))
        conv4_out = self.relu(self.conv4(conv3_out))
        conv5_out = self.relu(self.conv5(conv4_out))
        conv6_out = self.relu(self.conv6(conv5_out))
        conv7_out = self.conv7(conv6_out)
        x = conv7_out.squeeze()

        # Convert actions into action embeddings
        act = self.action_encoder(act)

        # Construct transformer sequence from conv outputs
        seq = torch.zeros((x.shape[0]*5, 128)).to(DEVICE)
        for i in range(x.shape[0]):
            idx = i * 5
            seq[idx] = x[i, :, 0, 0]
            seq[idx+1] = x[i, :, 0, 1]
            seq[idx+2] = x[i, :, 1, 0]
            seq[idx+3] = x[i, :, 1, 1]
            seq[idx+4] = act[i]
        plt.imshow(seq.detach().cpu().numpy().swapaxes(0,1))
        plt.show()
        seq = seq.unsqueeze(1)

        # Pass sequence through transformer
        # for _ in range(5):
        #     new_seq = self.encoder(seq)
        #     seq = torch.cat((seq[1:], new_seq[-1].unsqueeze(0)), dim=0)
        #     seq = self.transformer(seq, seq, memory_mask=self.transformer.generate_square_subsequent_mask(seq.shape[0]).to(DEVICE))
        # seq[-1] = act[-1]
        # trans_out = seq.squeeze()
        # seq = self.transformer(seq, seq)
        # seq[-1] = act[-1]
        # trans_out = seq.squeeze()
        seq = self.encoder(seq)
        seq[-1] = act[-1]
        trans_out = seq.squeeze()


        # Construct conv inputs for reconstruction
        deconv_in = torch.zeros((x.shape[0], 128, 2, 2)).to(DEVICE)
        for i in range(x.shape[0]):
            idx = (i * 5)
            deconv_in[i, :, 0, 0] = trans_out[idx] * trans_out[idx+4]
            deconv_in[i, :, 0, 1] = trans_out[idx+1] * trans_out[idx+4]
            deconv_in[i, :, 1, 0] = trans_out[idx+2] * trans_out[idx+4]
            deconv_in[i, :, 1, 1] = trans_out[idx+3] * trans_out[idx+4]

        # Deconvolve embeddings
        # out = self.deconv(deconv_in)

        act_emb = trans_out[torch.arange(x.shape[0])*5]
        smol_emb = self.big_to_smol(act_emb)

        deconv1_out = (self.relu(self.deconv1(deconv_in)) + conv7_out)
        deconv1_out *= act_emb.repeat(1, deconv1_out.shape[2] * deconv1_out.shape[3]).view(act_emb.shape[0], act_emb.shape[1], deconv1_out.shape[2], deconv1_out.shape[3])
        deconv2_out = (self.relu(self.deconv2(deconv1_out)) + conv6_out)
        deconv2_out *= act_emb.repeat(1, deconv2_out.shape[2] * deconv2_out.shape[3]).view(act_emb.shape[0], act_emb.shape[1], deconv2_out.shape[2], deconv2_out.shape[3])
        deconv3_out = (self.relu(self.deconv3(deconv2_out)) + conv5_out)
        deconv3_out *= act_emb.repeat(1, deconv3_out.shape[2] * deconv3_out.shape[3]).view(act_emb.shape[0], act_emb.shape[1], deconv3_out.shape[2], deconv3_out.shape[3])
        deconv4_out = (self.relu(self.deconv4(deconv3_out)) + conv4_out)
        deconv4_out *= act_emb.repeat(1, deconv4_out.shape[2] * deconv4_out.shape[3]).view(act_emb.shape[0], act_emb.shape[1], deconv4_out.shape[2], deconv4_out.shape[3])
        deconv5_out = (self.relu(self.deconv5(deconv4_out)) + conv3_out)
        deconv5_out *= act_emb.repeat(1, deconv5_out.shape[2] * deconv5_out.shape[3]).view(act_emb.shape[0], act_emb.shape[1], deconv5_out.shape[2], deconv5_out.shape[3])
        deconv6_out = (self.relu(self.deconv6(deconv5_out)) + conv2_out)
        deconv6_out *= smol_emb.repeat(1, deconv6_out.shape[2] * deconv6_out.shape[3]).view(smol_emb.shape[0], smol_emb.shape[1], deconv6_out.shape[2], deconv6_out.shape[3])
        deconv7_out = (self.relu(self.deconv7(deconv6_out)) + conv1_out)
        deconv7_out *= smol_emb.repeat(1, deconv7_out.shape[2] * deconv7_out.shape[3]).view(smol_emb.shape[0], smol_emb.shape[1], deconv7_out.shape[2], deconv7_out.shape[3])
        deconv8_out = (self.deconv8(deconv7_out))
        out = self.deconv9(deconv8_out)

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
        tidx = idx + random.randint(-
        0, 100040)
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
    model.load_state_dict(torch.load(PATH, map_location=DEVICE))
    model.eval()

env = gym.make("PongNoFrameskip-v4")
env = WarpFrame(env, width=84, height=84)
env = NoopResetEnv(env)
env = MaxAndSkipEnv(env)
env.reset()
step = 0

# Roll out env
for i in range(100):
    action = random.randint(0, 5)
    obs, rew, done, _ = env.step(action)
    DATA[step] = torch.tensor(obs.squeeze())
    ACTIONS[step] = action
    step += 1
    if done:
        env.reset()

# Roll out env'
for i in range(30):
    print("ROLLING OUT FRAME", i+1)
    out = model(DATA, ACTIONS)
    out = torch.argmax(out.permute(0,2,3,1), dim=3)
    DATA[:-1] = DATA[1:].clone()
    DATA[-1] = out[-1]
    out = out[-1]
    plt.imshow(out.squeeze().cpu().detach().numpy())
    plt.savefig('frames/frame' + str(i))

# Create the frames
frames = []
imgs = glob.glob("frames/*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('frames/frames.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=0)
