import torch
from torch import nn
import torch.nn functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
