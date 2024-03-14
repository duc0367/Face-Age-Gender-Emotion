import torch
import torch.nn as nn
import torch.optim as optim
from age_gender_model import AgeGenderNet


class AgeGenderTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AgeGenderNet()
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, dataloader, epochs: int):
        pass
