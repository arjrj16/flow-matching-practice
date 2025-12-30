import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch
import os

# Define the same model architecture as in main.py
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


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint"""
    # Create model with same architecture
    model = MLP(layers=5, channels=512)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded model from {checkpoint_path}")
    if 'step' in checkpoint:
        print(f"  Training step: {checkpoint['step']}")
    if 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.6f}")
    
    return model


def sample(model, num_samples=1000, steps=1000, device='cpu'):
    """Generate samples using the trained model"""
    model = model.to(device)
    
    # Start from random noise
    xt = torch.randn(num_samples, 2, device=device)
    
    print(f"\nGenerating {num_samples} samples with {steps} steps...")
    
    # Euler integration to solve the ODE
    with torch.no_grad():
        for i, t_val in enumerate(torch.linspace(0, 1, steps, device=device), start=1):
            t = t_val.expand(xt.size(0))
            pred = model(xt, t)
            dt = 1.0 / steps
            xt = xt + dt * pred
            
            # Print progress every 10% of steps
            if i % (steps // 10) == 0:
                print(f"  Step {i}/{steps} (t={t_val:.3f})")
    
    return xt.cpu().numpy()


def visualize_samples(samples, save_path=None):
    """Visualize the generated samples"""
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.6)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Generated Samples from Flow Matching Model')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Configuration
    checkpoint_path = "checkpoints/model_final.pt"  # Change this to use a different checkpoint
    num_samples = 1000
    steps = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        if os.path.exists("checkpoints"):
            for f in os.listdir("checkpoints"):
                if f.endswith(".pt"):
                    print(f"  - checkpoints/{f}")
        exit(1)
    
    # Load model
    model = load_model(checkpoint_path, device=device)
    
    # Generate samples
    samples = sample(model, num_samples=num_samples, steps=steps, device=device)
    
    # Visualize
    visualize_samples(samples, save_path="generated_samples.png")
    
    print(f"\nGenerated {len(samples)} samples!")
    print(f"Sample range: x=[{samples[:, 0].min():.2f}, {samples[:, 0].max():.2f}], "
          f"y=[{samples[:, 1].min():.2f}, {samples[:, 1].max():.2f}]")

