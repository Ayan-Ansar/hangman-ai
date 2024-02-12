import torch
from config import * 
import torch.nn as nn
from torch.nn import functional as f 



class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.token_embd = nn.Embedding(VOCAB_SIZE, N_EMBD) # B, T, C 
        self.output = nn.Linear(N_EMBD*BLOCK_SIZE, VOCAB_SIZE)

    def forward(self, x, target):
        B, T = x.shape
        x = self.token_embd(x) # B, T, C
        x = x.view(B, -1)
        logits = self.output(x) # B, vocab_size
        
        if target is None:
            loss = None
        else:
            loss = f.cross_entropy(logits, target)
            
        return logits, loss  





        