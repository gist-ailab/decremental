import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torch
from scipy import io as mat_io
from glob import glob

class CUB(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.train = train
                
        self._load_metadata()

        self.targets = [self.data.iloc[idx].target-1 for idx in range(len(self.data))]


    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                            names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                        sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                    sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'images', sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class Stanford_Cars(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, root, transform, train=True):
        self.root = root
        self.train = train
        self.loader = default_loader
        
        self.data = []
        self.target = []
                
        if self.train:
            meta_path = os.path.join(root, 'devkit', 'cars_train_annos.mat')
        else:
            meta_path = os.path.join(root, 'devkit', 'cars_test_annos_withlabels.mat')
            
        labels_meta = mat_io.loadmat(meta_path)
        self.transform = transform
        
        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if self.train:
                data_dir = os.path.join(root, 'train')
            else:
                data_dir = os.path.join(root, 'test')
                
            self.data.append(os.path.join(data_dir, img_[5][0]))
            self.target.append(img_[4][0][0])
        
        self.targets = [target-1 for target in self.target]

    def __getitem__(self, idx):
        image= self.loader(self.data[idx])

        if self.transform is not None:
            image = self.transform(image)
        
        target = torch.tensor(self.target[idx] - 1, dtype=torch.long)
        return image, target


    def __len__(self):
        return len(self.data)
