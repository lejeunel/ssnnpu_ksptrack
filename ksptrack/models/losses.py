import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class WeightedMSE(nn.Module):
    def __init__(self, weights, cuda=True):
        super(WeightedMSE, self).__init__()

        self.device = torch.device("cuda:0" if cuda \
                                   else "cpu")

        self.weights = weights

    def set_weights(self, weights):

        self.weights = weights

    def forward(self, y, y_true):

        weight_mask = torch.zeros_like(y_true).to(device=self.device)
        for c in self.weights.keys():
            w = torch.Tensor([self.weights[c]]).to(device=self.device)
            weight_mask += torch.where(
                y_true.eq(torch.Tensor([c]).to(device=self.device)),
                torch.ones_like(y_true, device=self.device) * w,
                torch.zeros_like(y_true, device=self.device))

        L = (((y - y_true) * weight_mask).pow(2)).mean()

        return L

class PriorMSE(nn.Module):
    def __init__(self, cuda=True):
        super(PriorMSE, self).__init__()

        self.device = torch.device("cuda" if cuda \
                                   else "cpu")

    def forward(self, y, y_true, prior):

        L = ((y - y_true).pow(2) * prior).mean()

        return L
