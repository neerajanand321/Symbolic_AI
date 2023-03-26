import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# PyTorch dataset class for training
class TrainDataset(Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
        
  def __len__(self):
      return len(self.dataset)
    
  def __getitem__(self, idx):
      data = self.dataset.iloc[idx]
      return torch.tensor([[float(data['func_tokens'])]]), torch.tensor(data['expnsn_token'])

# PyTorch dataset class for testing
class TestDataset(Dataset):
    def __init__(self, dataset):
      self.dataset = dataset
        
    def __len__(self):
      return len(self.dataset)
    
    def __getitem__(self, idx):
      data = self.dataset[idx]
      return torch.tensor([[float(data)]])

# class to predict given the function 

class Predict():
    def __init__(self, function, model, vocab_to_int, int_to_vocab):
        super(Predict, self).__init__()
        self.function = function
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = int_to_vocab
        self.model = model 

    def predict(self):
        inp = self.vocab_to_int[self.function]
        test_loader = DataLoader(TestDataset([inp]), batch_size=1)
        for i in test_loader:
            out = self.model(i)
        pred = out.argmax().item()
        pred = self.int_to_vocab[pred]
        return pred












