import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# reading the words from training file 
with open('words_250000_train.txt','r', encoding='utf-8') as f:
    text = f.read().splitlines() 

# mapping, encode and decode functions 
charset = sorted(list(set(''.join(text))))

STOI = {c:idx+2 for idx,c in enumerate(charset)}
ITOS = {idx+2:c for idx,c in enumerate(charset)}
STOI['.'] = 1 
ITOS[1] = '.'
STOI[''] = 0
ITOS[0] = ''

ENCODE = lambda x: [STOI[i] for i in x] 
DECODE = lambda x: ''.join([STOI[i] for i in x])

VOCAB_SIZE = len(STOI)
BLOCK_SIZE = max(len(word) for word in text)
BATCH_SIZE = 32

# Hyperparameters 
N_EMBD = 2
EPOCHS = 10000
LEARNING_RATE = 1e-2

