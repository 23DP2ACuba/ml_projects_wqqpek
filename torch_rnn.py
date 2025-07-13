import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    out, _ = self.rnn(x, h0)
    out = out.reshape(out.shape[0], -1)
    out = self.fc(out)
    return out

device = torch.device('cpu')
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
lr = 0.001
batch_size = 64
epochs = 1

train_dataset = datasets.MNIST(root="dataset/", train = True, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root="dataset/", train = False, transform = transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
  print(f"epoch: {epoch}>>", end= " ")
  for idx, (data, targets) in enumerate(train_loader):
    data = data.to(device).squeeze(1)
    target = targets.to(device)
    y_pred = model(data)
    loss = criterion(y_pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if idx % 100 == 0:
      print("|", end="")
  print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

def check_accuracy(loader, model):
  if loader.dataset.train:
    print("running on train dataset")
  else:
    print("running on test dataset")

  num_correct = 0
  num_samples = 0
  model.eval()
  with torch.no_grad():
    for x, y in loader:
      x = x.to(device)
      y = y.to(device)
      x = x.squeeze(1)
      y_pred = model(x)
      _, predictions = y_pred.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
    print(f"{num_correct}/{num_samples} with acc: {float(num_correct/num_samples) * 100:.2f}")

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
