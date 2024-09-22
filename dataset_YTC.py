import os
import random
from scipy.io import loadmat
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, split='train', shuffle=True, augmentation=False):
        super(CustomDataset, self).__init__()

        self.data_dir = data_dir
        self.split = split
        self.shuffle = shuffle
        self.augmentation = augmentation

        if split == 'train':
            self.data_folder = os.path.join(data_dir, 'train')
        elif split == 'test':
            self.data_folder = os.path.join(data_dir, 'test')
        elif split == 'train_pre':
            self.data_folder = os.path.join(data_dir, 'train_pre')
        else:
            raise ValueError("Invalid split. Use 'train' or 'train_pre' or 'test'.")

        self.file_list = os.listdir(self.data_folder)
        if shuffle:
            random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)
        data = loadmat(file_path)
        label = data['label']
        data = data['data']

        return {'data': data, 'label': label}