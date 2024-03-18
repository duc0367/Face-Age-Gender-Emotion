import torch
import torch.nn as nn
import torch.optim as optim
from emotion_model import EmotionNet
from tqdm import tqdm
import cv2
from torchvision import transforms
from dataset import map_idx_to_emotion, EmotionDataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

TEST_IMAGE = '../data/emotion/happy/10000.jpg'


def interpret(model, file_path: str, device=torch.device('cpu')):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Resize((48, 48))])
    image = transformer(image).to(device).unsqueeze(0)
    output = model(image)
    print(output.shape)
    output = torch.argmax(output, dim=1)
    print(output[0].item())
    return map_idx_to_emotion[output[0].item()]


class EmotionTrainer:
    def __init__(self, dataloader, epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.dataloader = dataloader
        self.epochs = epochs

    def train(self):
        losses = []
        for epoch in range(self.epochs):
            epoch_loss = []
            self.model.train()
            for idx, batch in enumerate(tqdm(self.dataloader)):
                image_batch, label_batch = batch['image'], batch['emotion']
                image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)
                print(label_batch)
                predictions = self.model(image_batch)
                print(predictions)
                loss = self.criterion(predictions, label_batch)
                epoch_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if idx + 1 % 100 == 0:
                    self.model.eval()
                    print(f"Epoch: {epoch}, batch: {idx}, loss: {loss.item():.4f}")
                    self.model.eval()
                    with torch.no_grad():
                        prediction = interpret(self.model, TEST_IMAGE, self.device)
                    print(f'Emotion prediction of image {TEST_IMAGE}: {prediction}')
                    self.model.train()
            losses.append(sum(epoch_loss) / len(epoch_loss))
        x_points = np.arange(0, len(losses))
        plt.plot(x_points, losses)
        plt.show()


if __name__ == "__main__":
    emotion_folder_path_ = "../data/emotion"
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Resize((48, 48)),
                                     transforms.RandomHorizontalFlip(0.3),
                                     transforms.RandomVerticalFlip(0.3)])
    dataset = EmotionDataSet(emotion_folder_path_, transform_)
    dataloader_ = DataLoader(dataset, batch_size=4,
                             shuffle=True, num_workers=4)

    trainer = EmotionTrainer(dataloader_, 1)
    trainer.train()
