import torch
import torch.nn.functional as F
import numpy as np 
import pickle 
from collections import Counter
import random 
from config import * 
from exception import AppException
from torch.nn.utils.rnn import pad_sequence 
import sys

random.seed(220)

def build_dataset(words):
    x = []
    y = []
    for word in words:
        encoded = torch.tensor(ENCODE(word))
        input = torch.ones_like(encoded)
        x.append(input) 
        sorted_letter_count = Counter(word).most_common()
        g = [0]*26
        prev = torch.clone(x[-1])
        while not torch.all(prev != 1):
            prev = torch.clone(x[-1])
            for letter, _ in sorted_letter_count:
                idx = ord(letter) - ord('a')
                if g[idx] != 1:
                    g[idx] = 1
                    g_in = STOI[letter]
                    break
            y.append(g_in)
            idxs = torch.where(encoded == g_in)[0]
            prev[idxs] = g_in 
            if torch.any(prev == 1):
                x.append(prev)
            
    # adding paddding to every input 
    x = pad_sequence(x, batch_first=True, padding_value=0)
    x = F.pad(x, (0, (BLOCK_SIZE - x.size()[1])), value=0)
    y = torch.tensor(y, dtype=torch.long)

    return x,y 
            
random.shuffle(text)
n1 = int(0.9*len(text))
n2 = int(0.95*len(text))

print('----------------------------------------------------------------')
Xtr, Ytr = build_dataset(text[:n1])
print('Successfully built training split : ', Xtr.shape, Ytr.shape)
print('----------------------------------------------------------------')
Xval, Yval = build_dataset(text[n1:n2])
print('Successfully built val split : ', Xval.shape, Yval.shape)
print('----------------------------------------------------------------')
Xts, Yts  = build_dataset(text[n2:])
print('Successfully built testing split : ', Xts.shape, Yts.shape, end='\n')



# dumping these files into pickle 

def dump_in_pkl(split):
    x, y = {'train' : (Xtr, Ytr),
            'val' : (Xval, Yval),
            'test' : (Xts, Yts)}[split]
    try:
        with open(f'data\X_{split}.pkl', 'wb') as f:
            pickle.dump(x, f)
    except Exception as e:
        raise AppException(e, sys)
            
    try:
        with open(f'data\Y_{split}.pkl', 'wb') as f:
            pickle.dump(y, f)
    except Exception as e:
        raise AppException(e, sys)        

dump_in_pkl('train')
dump_in_pkl('val')
dump_in_pkl('test') 
print('--------- Successfully created pickle files for datasets ----------')