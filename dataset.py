import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import cv2
from tqdm import tqdm


def Identity(x): return x


class FINNgerDataset(Dataset):
    """Hand Images dataset available at https://www.kaggle.com/koryakinp/fingers."""

    NUM_CLASSES = 6

    def __init__(self, data_dir, transform=Identity):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample, by default is Identity.
        """
        self.data_dir = data_dir
        self.transform = transform

        self.glob_path = glob.glob(data_dir)
        self.dataset = []
        for img_path in tqdm(self.glob_path, desc="Import data"):
            # Images are in the format <randomname>_<class>.png and here we are parsing the number from the class characters
            image_label = int(img_path[-6:-5])

            image = cv2.imread(img_path)

            self.dataset.append({'image': image, 'label': image_label})
        self.dataset = np.array(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Return in a good format for testing
        sample = self.dataset[idx]
        return (
            self.transform(sample['image']),
            sample['label'],
        )
