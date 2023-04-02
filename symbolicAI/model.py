import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the LSTM model
class LSTMModel(nn.Module):
  '''
  Implemenation of LSTM Model
  Input.shape = [batch_size, input_size, embedding_size]
  Output.shape = [batch_size, output_size]
  '''
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    # LSTM layer
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    # Fully connected layer 
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :]) # only take the last hidden state
    return out

# Define the Transformer model

class TransformerModel(nn.Module):
  '''
  Implementation of Transformer Model
  Input.shape = [batch_size, input_size, embedding_size]
  Output.shape = [batch_size, output_size]
  '''
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_heads, dropout):
    super(TransformerModel, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dropout = dropout
    
    self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
    self.encoder_layers = nn.ModuleList([
        nn.TransformerEncoderLayer(self.hidden_dim, self.num_heads, self.hidden_dim*4, self.dropout)
        for _ in range(self.num_layers)
    ])
    self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

  def forward(self, x):
    # x shape: (seq_length, batch_size, input_dim)
      
    # Embedding
    x = self.embedding(x)  # (seq_length, batch_size, hidden_dim)
      
    # Encoder layers
    for i in range(self.num_layers):
        x = self.encoder_layers[i](x)  # (seq_length, batch_size, hidden_dim)
      
    # Output layer
    x = x.mean(dim=0)  # (batch_size, hidden_dim)
    x = self.output_layer(x)  # (batch_size, output_dim)
      
    return x
  
  class seq2seqTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.encoder = nn.Embedding(input_dim, d_model)
        self.decoder = nn.Embedding(output_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, 
                                          dropout=dropout)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, src, trg):
        src = self.encoder(src)
        trg = self.decoder(trg)
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.size(0)).to(trg.device)
        output = self.transformer(src, trg, tgt_mask=trg_mask)
        output = self.fc(output)
        output = output.permute(1, 0, 2)
        return output
