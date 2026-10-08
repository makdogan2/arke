import torch
import torch.nn.functional as F

torch.manual_seed(42)

B, T, C = 1, 6, 32 #batch size, embedding size, number of characters
head_size = 16

x = torch.randn(B, T, C) #6 word random word

# Q, K, V matrices
key = torch.nn.Linear(C, head_size, bias=False)
query = torch.nn.Linear(C, head_size, bias=False)
value = torch.nn.Linear(C, head_size, bias=False)

k = key(x)   #(B, T, 16)
q = query(x) #(B, T, 16)
v = value(x) #(B, T, 16)

wei = q @ k.transpose(-2, -1) * head_size**-0.5 #(B, T, T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

out = wei @ v   # (B, T, 16)

print("weight matrix:")
print(wei[0].round(decimals=2))
print("output shape:", out.shape)