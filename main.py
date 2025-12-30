from calendar import c
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch
import tqdm
# create checkkerboard dataset

N = 1000 # number of points to sample
x_min, x_max = -4, 4
y_min, y_max = -4, 4
resolution = 100 # resolution of the grid

x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y) # returns 2 2d arrays, each of dimensions (y, x) - 
# it is kind of like when you do matrix multiplication and you align two matricies to see the shape except you dont * and +
# example:
# x = np.array([1, 2, 3])
# y = np.array([4, 5])
# X, Y = np.meshgrid(x, y)
# X contains x-coordinates:
# array([[1, 2, 3],
#        [1, 2, 3]])
# Y contains y-coordinates:
# array([[4, 4, 4],
#        [5, 5, 5]])
# Now (X[i,j], Y[i,j]) gives you the (x, y) coordinate at position (i, j)


length = 4
checkerboard = np.indices((length, length)).sum(axis=0) % 2
# np.indices((length, length)) returns a tuple of 2 arrays, each of dimensions (length, length)
# the first array contains the row indices and the second array contains the column indices
# the sum of the two arrays is the checkerboard pattern
# the % 2 operation makes the pattern alternate between 0 and 1
# array([[0, 1, 0, 1],   # 0%2=0, 1%2=1, 2%2=0, 3%2=1
#        [1, 0, 1, 0],   # 1%2=1, 2%2=0, 3%2=1, 4%2=0
#        [0, 1, 0, 1],   # 2%2=0, 3%2=1, 4%2=0, 5%2=1
#        [1, 0, 1, 0]])  # 3%2=1, 4%2=0, 5%2=1, 6%2=0

sampled_points = []
while len(sampled_points) < N:
    x_sample = np.random.uniform(x_min, x_max)
    y_sample = np.random.uniform(y_min, y_max)

    #convert to indices of checkerboard
    i = int((x_sample - x_min) / (x_max - x_min) * length)
    j = int((y_sample - y_min) / (y_max - y_min) * length)

    if checkerboard[j, i] == 1:
        sampled_points.append((x_sample, y_sample))

sampled_points = np.array(sampled_points)

# plt.scatter(sampled_points[:, 0], sampled_points[:, 1], s=1)
# plt.show()


# network:

class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ff = nn.Linear(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, channels_data=2, layers=5, channels=512, channels_t=512):
        super().__init__()
        self.channels_t = channels_t
        self.in_projection = nn.Linear(channels_data, channels)
        self.t_projection = nn.Linear(channels_t, channels)
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(layers)])
        self.out_projection = nn.Linear(channels, channels_data)

    # create sin/cos embeddings for timestep
    def gen_t_embedding(self, t, max_position=10000):
        t = t*max_position
        half_dim = self.channels_t //2
        emb = math.log(max_position) / (half_dim -1)
        emb = torch.arange(half_dim, device = t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.channels_t % 2 == 1:
            emb = nn.functional.pad(emb, (0,1), mode = 'constant')
        return emb

    def forward(self, x, t):
        x = self.in_projection(x)
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        x = x + t
        x = self.blocks(x)
        x = self.out_projection(x)
        return x

# define model:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = MLP(layers=5, channels=512)
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

# training loop:
data = torch.Tensor(sampled_points).to(device)
training_steps = 100_000
batch_size = 64
save_interval = 10000  # Save every 10k steps
checkpoint_dir = "checkpoints"
import os
os.makedirs(checkpoint_dir, exist_ok=True)

pbar = tqdm.tqdm(range(training_steps), desc = 'Training...')
losses = []
for i in pbar:
    x1 = data[torch.randint(data.size(0), (batch_size,))]
    x0 = torch.randn_like(x1)
    target = x1-x0
    t = torch.rand(x1.size(0), device=device)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    pred = model(xt, t)
    loss = ((target - pred) **2).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    pbar.set_postfix(loss=loss.item())
    losses.append(loss.item())
    
    # Save checkpoint periodically
    if (i + 1) % save_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{i+1}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'step': i + 1,
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")

# sampling:
# xt = torch.randn(1000, 2)
# steps = 1000
# for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
#     pred = model(xt, t.expand(xt.size(0)))
#     st = xt + (1 / steps) * pred

# Save final model
final_model_path = os.path.join(checkpoint_dir, "model_final.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'training_steps': training_steps,
    'final_loss': losses[-1] if losses else None,
}, final_model_path)
print(f"\nSaved final model to {final_model_path}")




