import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

class ANN(nn.Module):
  def __init__(self, input_dim=137, hidden_dim=256):
    super().__init__()
    self.lin1 = nn.Linear(input_dim,hidden_dim)
    self.lin2 = nn.Linear(hidden_dim,1)
    self.dropout = nn.Dropout(0.3)
  def forward(self, x):
    x = self.lin1(x)
    x = nn.functional.sigmoid(x)
    x = self.dropout(x)
    x = self.lin2(x)
    x = nn.functional.sigmoid(x)
    return x
  
model = ANN()
print(model)