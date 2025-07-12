"""
Fake News Detector
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.nn import functional as f
from datasets import Dataset
import torch.nn as nn
import pandas as pd
import torch

fake_path = 'gossipcop_fake.csv'
real_path = 'gossipcop_real.csv'

fake = pd.read_csv(fake_path)["title"]
real = pd.read_csv(real_path)["title"]

fake_df = pd.DataFrame({'text': fake, 'label': 1})
real_df = pd.DataFrame({'text': real, 'label': 0})

df = pd.concat([real_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

