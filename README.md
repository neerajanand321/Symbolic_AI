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
