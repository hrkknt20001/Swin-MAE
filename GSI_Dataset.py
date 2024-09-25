import torch
import glob
import os
from PIL import Image

class GSI_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform = None):
        self.transform = transform
        self.dataset = glob.glob( os.path.join(path, '*/org286/*.png'))
        self.datanum = len(self.dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = Image.open(self.dataset[idx])
        if self.transform:
            out_data = self.transform(out_data)

        return out_data, 0
