import matplotlib.pyplot as plt
import numpy as np
import random


x = np.array([[1, 2, 3, 4, 5, 6], [3, 3, 3, 3, 3, 3]]).T
y = np.array([4, 5, 6, 7, 8, 9])

class LR:
  def __init__(self, x, y, lr):
      self.x = x
      self.y = y
      self.b = 0
      self.w = np.zeros(x.shape[1])
      self.lr = lr
      self.m = x.shape[0]

  def predict(self):
        self.pred = np.dot(self.x, self.w) + self.b

  def grad(self):
      err = self.pred - self.y
      self.wrt_w = (2 / self.m) * np.dot(self.x.T, err)
      self.wrt_b = (2 / self.m) * np.sum(err)

  def MSE(self):
    s = 0
    for i in range(len(self.y)):
      s += (self.pred[i] - self.y[i])**2
    return s

  def update(self):
    self.w = self.w - self.lr * self.wrt_w
    self.b = self.b - self.lr * self.wrt_b

  def r2score(self):
    r2 = 1
    RSS = 0
    TSS = 0
    for i in range(len(self.y)):
      RSS += (self.y[i] - self.pred[i])**2

    for i in range(len(self.y)):
      TSS += (self.y[i] - self.y.mean())**2
    return r2 - RSS/TSS

lr = LR(x, y, lr=0.01)
for i in range(1000):
  lr.predict()
  lr.grad()
  lr.update()


print(f"w = {lr.w}, b = {lr.b}")
print(f"R² score = {lr.r2score():.4f}, loss: {lr.MSE()}")
