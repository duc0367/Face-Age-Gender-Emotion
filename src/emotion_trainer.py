import torch
import torch.nn as nn
import torch.optim as optim
from emotion_model import EmotionNet


class EmotionTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionNet()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, dataloader, epochs):
        pass
