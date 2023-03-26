from sympy import *
import pandas as pd
import numpy as np
import nltk
from collections import Counter

# Class to create the dataset 
class Data():
    def __init__(self, order, types=[]):
        super().__init__()
        self.order = order # Upto which order we want the expansion
        self.types = types # List of the function for which we want expansion
        self.vocab_to_int = None # Dict which store int value corres to tokens in vocab
        self.int_to_vocab = None # Dict which store tokens corres to int value
        
    def generate(self):
        df = pd.DataFrame()
        function = []
        expansion = []
        x = symbols('x')
        for i in self.types:
            function.append(str(i))
            # Generating expansion for each function
            expnsn = i.series(x, 0, self.order+1).removeO()
            expansion.append(str(expnsn))
        df['function'] = function
        df['expansion'] = expansion
        # Dataframe which contain functions and their corresponding expansion
        df = df.sample(frac=1, random_state=2023, ignore_index=True)
        return df
    
    def tokenize(self, df):
        # Tokenize the dataset
        tokens = []
        for i in range(len(df)):
            tokens += [df['function'][i], df['expansion'][i]]

        # Create a vocabulary of tokens
        counter = Counter(tokens)
        vocab = sorted(counter, key=counter.get, reverse=True)
        self.vocab_to_int = {token: i for i, token in enumerate(vocab, 1)}
        self.int_to_vocab = {i: token for token, i in self.vocab_to_int.items()}

        # DataFrame which contain the tokens of functions and the corresponding token of expansion
        data_tokens = pd.DataFrame()

        func_tokens = []
        expnsn_tokens = []

        for i in range(len(df)):
            func_tokens.append(self.vocab_to_int[df['function'][i]])
            expnsn_tokens.append(self.vocab_to_int[df['expansion'][i]])

        data_tokens['func_tokens'] = func_tokens
        data_tokens['expnsn_token'] = expnsn_tokens

        return data_tokens

    def get_tokens_dict(self):
      # This function is used to get the token and vocab dict
      return [self.vocab_to_int, self.int_to_vocab]

