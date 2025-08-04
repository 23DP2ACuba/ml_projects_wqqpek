import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super.__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_dim_2img = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU()
        
        
    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        
        return mu, sigma
    
    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_dim_2img(h))
    
    def forward(self, x):
        mu, sigma= self.encode(x)
        eps = torch.rand_like(sigma)
        z_rep = mu + sigma * eps
        x_reconst = self.decode(z_rep)
        return x_reconst, mu, sigma
    
# if __name__ == "__main__":
#     x = torch.randn(4, 28*28)
#     vae = VAE(input_dim=784)
#     print((vae(x))[0].shape)
    