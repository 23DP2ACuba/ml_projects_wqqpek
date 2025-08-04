import torch
import torchvision.datasets as datasets
from tqdn import tqdn
from torch import nn, optim
from vae_model import VAE
from torchvision import transforms
from torchvision.utils import save_image

device = torch.device("cpu")

INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4

dataset = datasets.mnist(root="dataset/", train = True, transforms=transforms.ToTensor(),)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(INPUT_DIM, H_DIM, Z_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

loss_fn = nn.BCELoss(reduction="sum")

for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i, (x, _) in loop:
        x = x.to(device).view(x.shape[0], INPUT_DIM)
        x_reconst, mu, sigma = model(x)
        
        reconst_loss = loss_fn(x_reconst, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
        
mode = model.to("cpu")
def inference(digit, num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break
        
    encodings_digit = []
    
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.rand_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28) 
        save_image(out, f"generated_{digit}_ex{example}.png")           
        
for digit in range(10):
    inference(digit, num_examples=1)
            