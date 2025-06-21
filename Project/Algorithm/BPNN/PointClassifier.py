from cProfile import label
from filecmp import cmp
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

n_points = 100
torch.manual_seed(1000)
data = torch.rand(n_points,2)*2 - 1
labels = (data.norm(dim=1) > 0.7).float().unsqueeze(1)
class PointClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,20)
        self.layer2 = nn.Linear(20,1)
        
    def forward(self,x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
        
model = PointClassifier()
loss_fn = nn.BCELoss() # Binary Cross Entropy
optimizer = optim.AdamW(model.parameters(),lr = 0.1)
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(data)
    loss = loss_fn(predictions,labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch:{epoch}, Loss:{loss.item():.4f}")
