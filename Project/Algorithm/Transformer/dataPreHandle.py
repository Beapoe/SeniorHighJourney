import torch
import os

if "ADreamofRedMansions.txt" not in os.getcwd(): 
    os.chdir("./Project/Algorithm/Transformer")
with open("ADreamofRedMansions.txt","+r",encoding="utf-8") as file:
    originTxt = file.read()

chars = sorted(list(set(originTxt)))
str2token = {idx:s for s,idx in enumerate(chars)}
str2token.popitem()
token2str = {s:idx for s,idx in enumerate(chars)}
token2str.popitem()

encode = lambda str:[str2token[c] for c in str]
decode = lambda list:"".join([token2str[idx] for idx in list])
