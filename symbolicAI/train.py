import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMModel, TransformerModel

# Trainer class to train the model
class Train(nn.Module):
  def __init__(self, epoch, batch_size, input_size, hidden_size, num_layers, output_size):
    super(Train, self).__init__()
    self.epoch = epoch
    self.batch_size = batch_size
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.model = None
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def run(self, dataloader, model_type):
    self.model_type = model_type # Select model between LSTM and transformer
    if self.model_type == "LSTM":
      self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, self.output_size)
    if self.model_type == "Transformer":
      self.model = TransformerModel(self.input_size, self.hidden_size, self.num_layers, 
      self.output_size, num_heads=8, dropout=0.1)
    # Exporting Model to device
    self.model = self.model.to(self.device) 
    # Optimizer and Loss function
    optimizer = optim.Adam(self.model.parameters())
    criterion = nn.CrossEntropyLoss(reduction="sum")
    
    # Training
    for epoch in range(self.epoch):
      self.model.train() # Setting to train mode
      epoch_loss = 0
      for input_seq, target in dataloader:
        input_seq = input_seq.to(self.device)
        target = target.to(self.device)

        # Empty the optimizer dict before start of each epoch
        optimizer.zero_grad()
        output = self.model(input_seq)

        # calculating loss for backward pass
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss/37

      if (epoch+1)%1000 == 0 or epoch == 0:
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}')
    	
  def get_model(self):
    # return trained model so that can be used for testing
    model = self.model.to("cpu")
    return model



