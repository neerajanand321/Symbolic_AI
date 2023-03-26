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
  '''
  Arguments :
   -> function = mathematical function for which we want expansion eg; "sin(x)"
   -> model = Trained model
   -> vocab_to_int = dictionary containing int corresponding to each token
   -> int_to_vocab = dictionary containg token corresponding to each target int
  '''
  def __init__(self, function, model, vocab_to_int, int_to_vocab):
    super(Predict, self).__init__()
    self.function = function
    self.vocab_to_int = vocab_to_int
    self.int_to_vocab = int_to_vocab
    self.model = model 

  def predict(self):
    '''
    Return : Predicted expansion eg: '15*x**4 + 10*x**3 + 6*x**2 + 3*x + 1'
    '''
    inp = self.vocab_to_int[self.function]
    test_loader = DataLoader(TestDataset([inp]), batch_size=1)
    for i in test_loader:
        out = self.model(i)
    pred = out.argmax().item()
    pred = self.int_to_vocab[pred]
    return pred












