import numpy as np
import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader



class Perceptron(nn.Module):

    """
    contains multiple regression techniques on multimodal data
    """


    def __init__(self, audio, text, labels):
        self.audioInput = audio
        self.textInput = text
        self.labels = labels
        self.theta = np.zeros((2,))

        # combines audio and text input into a single feature vector
        # audio and text inputs are probalilities of each categeory
        self.features = np.vstack((self.audioInput, self.textInput)).T
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(12, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
        )






