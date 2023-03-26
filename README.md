# Symbolic Calculation Project

This project involves three tasks: data preprocessing, the use of an LSTM model, and the use of a transformer model.

## Common Task 1: Dataset Preprocessing

We have to use Sympy or Mathematica to generate datasets of functions with their Taylor expansions up the fourth order and to Tokenize the dataset.

- Created `dataset.py` in **symbolicAI** folder 
- Defined class `Data` in `dataset.py` which contain functions `generate` to generate the dataset using Sympy, `tokenize` to tokenize the dataset and `get_token_dict` to get the dictionary of tokens

## Common Task 2: Use LSTM Model

We have to train an LSTM model to learn the Taylor expansion of each function.

- Created `model.py` in **symbolicAI** folder
- Defined `LSTMModel` in `model.py` 

## Specific Task 3: Use Transformer model

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
move to the directory
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

