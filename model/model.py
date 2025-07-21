from torch.utils.data import DataLoader, TensorDataset
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import nltk
import json
import os

nltk.download("punkt_tab")

class Model(nn.Module):
    def __init__(self, inp_size, out_size, hidden_size=64, dropout=0.5):
        super().__init__()
        
        hidden_size_1 = hidden_size
        hidden_size_2 = hidden_size*2
        
        self.fc1 = nn.Linear(inp_size, hidden_size_2)
        self.fc2 = nn.Linear(hidden_size_2, hidden_size_1)
        self.fc3 = nn.Linear(hidden_size_1, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = dropout(self.relu(self.fc1(x)))
        x = dropout(self.relu(self.fc2(x)))
        return self.fc3(x)
        
        
class Assistant:
    def __init__(self, data_path, function_mappings=None):
        self.model = None
        self.data_path = data_path
        
        self.doc = []
        self.intents = []
        self.vocab = []
        self.responses = {}
        
        self.fn_mappings = function_mappings
        
        self.x, self.y = None, None
    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        
        return tokens
    
    def bag_of_words(self, tokens, vocab):
        return [i if token in tokens else 0 for token in vocab ]
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()
         
        if os.path.exists(self.data_path):
            with open(self.data_path, "r") as f:
                intents_data = json.load(f)
        
            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.responses[intent["tag"]] = intent["responses"]
                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocab.extend(pattern_words)
                    self.doc.append((pattern_words, intent["tag"]))
                self.vocab = sorted(set(self.vocab))  
                
    def prepare_data(self):
        bags = []
        indices = []
        
        for document in self.doc:
            words = document[0]
            bag = self.bag_of_words(words, self,vocab)
                
