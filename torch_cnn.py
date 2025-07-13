import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NN, self).__init__()
    self.fc1 = nn.Linear(input_size, 50)
    self.fc2 = nn.Linear(50, num_classes)

  def forward(self, x):
    x = f.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class CNN(nn.Module):
  def __init__(self, in_channels=1, n_classes=10):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=(1,1), stride=(1,1))
    self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2,2))
    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1,1), stride=(1,1))
    self.fc1 = nn.Linear(16*7*7, num_classes)
  def forward(self, x):
    x = f.relu(self.conv1(x))
    x = self.pool(x)
    x = f.relu(self.conv2(x))
    x = self.pool(x)
    x = x.reshape(x.shape[0], -1)
    return self.fc1(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 784
num_classes = 10
lr = 0.001
batch_size = 64
epochs = 1

train_dataset = datasets.MNIST(root="dataset/", train = True, download=True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_dataset = datasets.MNIST(root="dataset/", train = False, transform = transforms.ToTensor())
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
  for idx, (data, targets) in enumerate(train_loader):
    data = data.to(device)
    target = targets.to(device)

    data = data.reshape(data.shape[0], -1)
    y_pred = model(data)
    loss = criterion(y_pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



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
      x = x.reshape(x.shape[0], -1)
      y_pred = model(x)
      _, predictions = y_pred.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
    print(f"{num_correct}/{num_samples} with acc: {float(num_correct/num_samples) * 100:.2f}")

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
