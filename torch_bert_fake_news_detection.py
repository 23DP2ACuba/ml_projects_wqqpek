"""
Fake News Detector
"""
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.nn import functional as f
import torch.nn as nn
import pandas as pd
import torch


N_SEGMENTS = 3
MAX_LEN = 512
EMBEDDING_DIM = 768
DROPOUT = 0.1
ATTN_HEADS = 4
N_LAYERS = 12
LR = 2e-5
EPOCHS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fake_path = 'gossipcop_fake.csv'
real_path = 'gossipcop_real.csv'

fake = pd.read_csv(fake_path)["title"]
real = pd.read_csv(real_path)["title"]

fake_df = pd.DataFrame({'text': fake, 'label': 1})
real_df = pd.DataFrame({'text': real, 'label': 0})

df = pd.concat([real_df, fake_df], ignore_index=True)

train_dataset, val_dataset = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(data):
    tokens = tokenizer(
        list(data["text"]),
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt"
    )
    label = torch.tensor(data["label"].values)
    return TensorDataset(tokens["input_ids"], tokens['attention_mask'], label)

train_dataset = tokenize(train_dataset)
val_dataset = tokenize(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

VOCAB_SIZE = len(tokenizer)


class BERTEmbedding(nn.Module):
  def __init__(self, vocab_size, n_seg, max_len, embed_dim, dropout):
    super().__init__()
    self.token_emb = nn.Embedding(vocab_size, embed_dim)
    self.seg_emb = nn.Embedding(n_seg, embed_dim)
    self.pos_emb = nn.Embedding(max_len, embed_dim)
    self.dropout = nn.Dropout(dropout)
    self.pos_inp = torch.tensor([i for i in range(max_len)],)

  def forward(self, seq, seg):
    seq_len = seq.size(1)
    pos_ids = torch.arange(seq_len, device = seq.device).unsqueeze(0).expand(seq.size(0), seq_len)
    embeddings =  self.token_emb(seq) + self.seg_emb(seg) + self.pos_emb(self.pos_inp)
    return self.dropout(embeddings)

class BERT(nn.Module):
  def __init__(self, vocab_size, n_segments, max_len, embd_dim, n_layers, attn_heads, dropout):
    super().__init__()
    self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embd_dim,dropout)
    self.encoder_layer = nn.TransformerEncoderLayer(embd_dim, attn_heads, embd_dim*4)
    self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)
    self.cls_head = nn.Linear(embd_dim, 2)

  def forward(self, seq, seg):
    x = self.embedding(seq, seg)
    x = x.permute(1, 0, 2)
    x = self.encoder_block(x)
    cls_token = x[0]
    return self.cls_head(cls_token)



bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBEDDING_DIM, N_LAYERS, ATTN_HEADS, DROPOUT).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(bert.parameters(), lr=LR)

bert.train()
for epoch in range(EPOCHS):
  total_loss = 0

  for batch in train_loader:
    input_ids = batch[0].to(device)
    attention_mask = batch[1].to(device)
    label = batch[2].to(device)

    seg_ids = torch.zeros_like(input_ids)
    y_pred = bert(input_ids, seg_ids)
    loss = criterion(y_pred, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss

  print(f"Epoch: {epoch}, Loss: {total_loss}")
