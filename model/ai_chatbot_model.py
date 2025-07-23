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
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
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
    
    def bag_of_words(self, tokens):
        return [1 if word in tokens else 0 for word in self.vocab]
    
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
            bag = self.bag_of_words(words)
            
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)
            
        self.x = np.array(bags)
        self.y = np.array(indices)     
        
    def train_model(self, batch_size, lr, epochs):
        x_tensor = torch.tensor(self.x, dtype = torch.float32)
        y_tensor = torch.tensor(self.y, dtype = torch.long)
        
        ds = TensorDataset(x_tensor, y_tensor)

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        self.model = Model(self.x.shape[1], len(self.intents))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
         
        for epoch in range(epochs):
            losses = 0.0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                
                loss = criterion(out, batch_y)
                loss.backward()
                
                optimizer.step()
                
                losses += loss
            print(f"Epoch: {epoch+1}, Loss: {losses/len(loader):4f}")
            
            
            
    def save_model(self, model_path, dim_path):
        torch.save(self.model.state_dict(), model_path)
        
        with open(dim_path, "w") as f:
            json.dump({
                        "input_size": self.x.shape[1], 
                        "output_size": len(self.intents)
                       }, f)
            
    def load_model(self, model_path, dim_path):
        with open(dim_path, "w") as f:
            dim = json.load(f)
            
        self.model = Model(dim["input_size"], dim["output_size"])
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        
    def process_message(self, input_message):
        words =self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        
        bag_tensor = torch.tensor([bag], dtype = torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(bag_tensor)
        
        predicted_class_index = torch.argmax(pred, dim=1).item()
        
        predicted_intent = self.intents[predicted_class_index]
        
        if self.fn_mappings:
            if predicted_intent in  self.fn_mappings:
                self.fn_mappings[predicted_intent]()
        
        if self.responses[predicted_intent]:
            return random.choice(self.responses[predicted_intent])
        else:
            return None

def get_stocks():
    stocks = ["AAPL", "META", "NVDA", "GS", "MSFT"]
    return random.sample(stocks, 3)

if __name__ == "__main__":
    assistant = Assistant("intents.json", function_mappings = {"stocks": get_stocks})
        
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs = 100)
    assistant.save_model("chatbot_model.pth", "dimensions.json")

    while True:
      message = input("Enter your message:")

      if message == "/quit":
        break

      print(assistant.process_message(message))
