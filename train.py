import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from model import *
from config import *
from utils import * 
import matplotlib.pyplot as plt


model = Model()
m = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
num_params = sum(p.numel() for p in model.parameters(recurse=True))

print(f'{DEVICE = }')
print('--------------------------------')
print(f'{num_params = }')
print('--------------------------------')

@torch.no_grad()
def split_loss():
    model.eval()
    out = {}
    for split in ['train','val']:
        losses = torch.zeros(200)
        for k in range(200):
            xb, yb = get_batch(split)
            logits, loss = model(xb,yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


lossi = []

for epoch in range(EPOCHS):
    if epoch % 1000 == 0:
        loss = split_loss()
        print(f"step {epoch}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # for name, p in model.named_parameters():
    #     p.retain_grad = True
    optimizer.zero_grad(set_to_none=True)
    lossi.append(loss.item())
    loss.backward()
    optimizer.step()
    # with torch.no_grad(): 
    #     ud.append([(learning_rate*p.grad.std() / p.data.std()).log10().item() for _, p in model.named_parameters()]) 

loss = split_loss()
print(f"step {EPOCHS}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")

plt.plot(lossi)
plt.show()
