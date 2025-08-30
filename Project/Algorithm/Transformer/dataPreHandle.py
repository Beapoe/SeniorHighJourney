import torch
import os

with open("ADreamofRedMansions.txt","+r",encoding="utf-8") as file:
    originTxt = file.read()

chars = sorted(list(set(originTxt)))
str2token = {idx:s for s,idx in enumerate(chars)}
token2str = {s:idx for s,idx in enumerate(chars)}
print(str2token)

encode = lambda str:[str2token[c] for c in str]
decode = lambda list:"".join([token2str[idx] for idx in list])

data = torch.tensor(encode(originTxt),dtype=torch.long)
print(data)