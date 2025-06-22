import torch
import torch.nn as nn
from torch.nn import functional as f

# ------- config -------
batch_size = 64
block_size = 256
max_iters = 6500
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200
torch.manual_seed(1337)
train = True
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ----------------------

with open('input.txt', "r", encoding = "utf-8") as text_inp:
  text = text_inp.read()

chars = sorted(list(set(text)))
vocab_len = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
  return [stoi[c] for c in s]
def decode(l):
  return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(train):
  data = train_data if train else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
  y = torch.stack([data[i+1: i+block_size+1] for i in ix]).to(device)
  return x, y


class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    b, t, c = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
    wei = f.softmax(wei, dim=-1)
    v = self.value(x)
    out = wei @ v
    return out

class MultiHead(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(n_embd * 4, n_embd),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_head, n_embd):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHead(n_head, head_size)
    self.fwd = FeedForward(n_embd)
    self.l1 = nn.LayerNorm(n_embd)
    self.l2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa(self.l1(x))
    x = x + self.fwd(self.l2(x))
    return x
class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_len, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.block = nn.Sequential(*[Block(n_head, n_embd) for _ in range(n_layer)])
    self.ln = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_len)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.2)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.2)

  def forward(self, idx, target=None):
    b, t = idx.shape
    tok_emd = self.token_embedding_table(idx)
    pos_emd = self.position_embedding_table(torch.arange(t, device=device))
    x = tok_emd + pos_emd
    x = self.block(x)
    x = self.ln(x)
    logits = self.lm_head(x)

    if target is not None:
      b, t, c = logits.shape
      logits = logits.view(b * t, c)
      target = target.view(-1)
      loss = f.cross_entropy(logits, target)

    else:
      loss = None

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = f.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx

model = GPT().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in [True, False]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(losses)

    xb, yb = get_batch(train)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(idx = context, max_new_tokens=1000)[0].tolist()))
