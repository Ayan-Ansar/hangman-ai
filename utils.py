import torch
import pickle
from exception import AppException 
import sys
from config import *

def load_split(split):
    try:
        with open(f'data\X_{split}.pkl', 'rb') as f:
            X = pickle.load(f)
    
    except Exception as e:
        raise AppException(e, sys)
    
    try:
        with open(f'data\Y_{split}.pkl', 'rb') as f:
            Y = pickle.load(f)
    
    except Exception as e:
        raise AppException(e, sys)
    
    return X, Y

def get_batch(split):
    X, Y = {'train': load_split('train'),
            'val' : load_split('val')}[split]
    ix = torch.randint(0, X.shape[0], (BATCH_SIZE,))
    xb, yb = X[ix].to(DEVICE), Y[ix].to(DEVICE)
    return xb, yb


