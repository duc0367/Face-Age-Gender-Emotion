import torch
from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms


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


if __name__ == "__main__":
    folder_path_ = "../data/age-gender"
    transform_ = transforms.Compose([transforms.ToTensor()])
    dataset = AgeGenderDataset(folder_path_, transform_)
