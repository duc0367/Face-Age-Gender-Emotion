import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
from enum import Enum


class Emotion(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    NEUTRAL = 4
    SAD = 5
    SURPRISE = 6


map_emotion_to_idx = {
    'angry': Emotion.ANGRY,
    'disgust': Emotion.DISGUST,
    'fear': Emotion.FEAR,
    'happy': Emotion.HAPPY,
    'neutral': Emotion.NEUTRAL,
    'sad': Emotion.SAD,
    'surprise': Emotion.SURPRISE
}


class AgeGenderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform
        self.files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        image = cv2.imread(os.path.join(self.folder_path, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = filename.split('_')
        age = int(data[0])
        gender = int(data[1])
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'age': age, 'gender': gender}


class EmotionDataLoader(Dataset):
    def __init__(self, folder_path: str, transform=None):
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform
        self.filenames = []
        emotion_folders = [folder for folder in os.listdir(os.path.join(self.folder_path))
                           if os.path.isdir(os.path.join(self.folder_path, folder))]
        for emotion in emotion_folders:
            emotion_folder = os.path.join(folder_path, emotion)
            files = [(os.path.join(emotion_folder, file), map_emotion_to_idx[emotion])
                     for file in os.listdir(emotion_folder) if os.path.isfile(os.path.join(emotion_folder, file))]
            self.filenames.extend(files)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename, label = self.filenames[idx]
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'emotion': label}


if __name__ == "__main__":
    age_gender_folder_path_ = "../data/age-gender"
    emotion_folder_path_ = "../data/emotion"
    transform_ = transforms.Compose([transforms.ToTensor(), transforms.Resize((200, 200)),
                                     transforms.RandomHorizontalFlip(0.3),
                                     transforms.RandomVerticalFlip(0.3)])

    transform2_ = transforms.Compose([transforms.ToTensor(), transforms.Resize((48, 48)),
                                      transforms.RandomHorizontalFlip(0.3),
                                      transforms.RandomVerticalFlip(0.3)])
    # dataset = AgeGenderDataset(folder_path_, transform_)
    emotions = EmotionDataLoader(emotion_folder_path_, transform2_)
    print(emotions.__getitem__(1))
