import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from age_gender_model import AgeGenderNet
from tqdm import tqdm
import cv2
from torchvision import transforms
from dataset import AgeGenderDataset
import matplotlib.pyplot as plt
import numpy as np

TEST_IMAGE = '../data/age-gender/9_1_4_20170103213057382.jpg.chip.jpg'


def interpret(model, file_path: str, device) -> tuple:
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize((200, 200))])
    image = transformer(image).to(device).unsqueeze(0)
    predictions = model(image)
    return predictions[0]


class AgeGenderTrainer:
    def __init__(self, dataloader, epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AgeGenderNet()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.epochs = epochs
        self.dataloader = dataloader

    def train(self):
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = []
            self.model.train()
            for idx, batch in enumerate(tqdm(self.dataloader)):
                image_batch, gender_batch, age_batch = batch['image'], batch['gender'], batch['age']
                image_batch, gender_batch, age_batch = (
                    image_batch.to(self.device), gender_batch.to(self.device), age_batch.to(self.device))
                gender_batch = torch.unsqueeze(gender_batch, dim=-1)
                age_batch = torch.unsqueeze(age_batch, dim=-1)
                gender_predictions, age_predictions = self.model(image_batch)
                gender_loss = self.criterion(gender_predictions, gender_batch)
                age_loss = self.criterion(age_predictions, age_batch)
                loss = gender_loss + age_loss
                epoch_loss.append(loss.detach().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if idx + 1 % 100 == 0:
                    print(f"Epoch: {epoch}, loss: {loss.item():.4f}")
                    self.model.eval()
                    with torch.no_grad():
                        (gender, age) = interpret(self.model, TEST_IMAGE, self.device)
                    print(f"Image {TEST_IMAGE}: Age: {age}, Gender: {gender}")
                    self.model.train()
            losses.append(sum(epoch_loss)/len(epoch_loss))
        x_points = np.arange(0, len(losses))
        plt.plot(x_points, losses)
        plt.show()


if __name__ == "__main__":
    age_gender_folder_path_ = "../data/age-gender"
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Resize((200, 200)),
                                     transforms.RandomHorizontalFlip(0.3),
                                     transforms.RandomVerticalFlip(0.3)])
    dataset = AgeGenderDataset(age_gender_folder_path_, transform_)
    dataloader_ = DataLoader(dataset, batch_size=4,
                             shuffle=True, num_workers=4)

    trainer = AgeGenderTrainer(dataloader_, 1)
    trainer.train()
