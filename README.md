# Symbolic Calculation Project

This project involves three tasks: dataset creation (preprocessing), the use of an LSTM model, and the use of a transformer model.
<p align="center">
  <img src="https://user-images.githubusercontent.com/78193865/227809564-1dd1b7a6-f70e-496d-b327-6b8d5ca7171e.png" width="350" title="hover text">
</p>

### Common Task 1: Dataset Preprocessing

We have to use Sympy or Mathematica to generate datasets of functions with their Taylor expansions up the fourth order and to Tokenize the dataset.

- Created `dataset.py` in **symbolicAI** folder 
- Defined class `Data` in `dataset.py` which contain functions `generate` to generate the dataset using Sympy, `tokenize` to tokenize the dataset and `get_token_dict` to get the dictionary of tokens

### Common Task 2: Use LSTM Model

**Long Short-Term Memory (LSTM)** networks are a type of recurrent neural network capable of learning order dependence in sequence prediction problems.
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/LSTM_Cell.svg" width="350" title="hover text">
</p>


We have to train an LSTM model to learn the Taylor expansion of each function.

- Created `model.py` in **symbolicAI** folder
- Defined `LSTMModel` in `model.py` 

### Specific Task 3: Use Transformer model
A **transformer** is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data.
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png" width="350" title="hover text">
</p>

We have to train a Transformer  model to learn the Taylor expansion of each function.

- Defined `TransformerModel` in `model.py`

**`train.py` contains `Train` class which is used to train the model and to get the trained model for prediction, `utils.py` contain class `TrainDataset` to create PyTorch dataset for training, `TestDataset` to create PyTorch dataset for testing and `Predict` class to get the prediction provided any function**

#### `Symbolic_AI.ipynb` demonstrates the creation of dataset, tokenization of dataset, training of LSTM and Transformer model and the Prediction of model on various mathematical functions.

## Usage

#### 1) Setup
Clone the repository 
```bash
$ git clone https://github.com/neerajanand321/Symbolic_AI.git
```
Move to the directory
```bash
cd Symbolic_AI/symbolicAI
```

#### 2) Generate Dataset 
```python
from dataset import data
from sympy import *

order = 4
types = [sin(x), cos(x), exp(x), log(1+x)]

data = Data(order, types) # Create data object
data_df = data.generate()
""" returns Dataframe containing functions and their corresponding expansion"""
```

#### 3) Tokenize Dataset
```python
df_tokenized = data.tokenize(data_df) 
""" 
Input = Data to be tokenized 
returns Dataframe containing token of functions and token of expansions
"""
```

#### 4) Dataset class
```python
from utils import TrainDataset, TestDaatset
data = TrainDataset(df_tokenized)
"""returns PyTorch Dataset for training"""
```

#### 5) Training
```python
# Hyperparameter
epoch = 15000
batch_size = 1
input_size = 1
hidden_size = 64
num_layers = 2
output_size = len(vocab_to_int) + 1
model_type = "LSTM" # Change it to "Transformer" to train the transformer model

# Triaining the model
from train import Train
from torch.utils.data import DataLoader
train_obj = Train(epoch, batch_size, input_size, hidden_size, 
                  num_layers, output_size)
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
losses = train_obj.run(train_loader, model_type)
"""returns list of epoch loss"""
```

#### 6) Prediction
```python
example = "sqrt(x + 1)"
model = train_obj.get_model()
pred_obj = Predict(example, model, vocab_to_int, int_to_vocab)
ans = pred_obj.predict()
ans
"""returns expansion of given function eg: '-5*x**4/128 + x**3/16 - x**2/8 + x/2 + 1'"""
```

## Dependencies

To run the notebooks, you will need to have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- sympy
- NumPy
- Pandas
- Matplotlib
- PyTorch

You can install these dependencies using pip or conda.
