import torch
import glob
import os
from PIL import Image
import albumentations as A
import cv2
import numpy as np

def get_augmentation(name):
    if name == 'Original':
        train_transform = [
            A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            A.ToTensorV2()
        ]
    elif name == 'RandomCrop':
        train_transform = [
            A.RandomCrop(height=286, width=286),
            A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            A.ToTensorV2()
        ]
    elif name == 'RandomRotate':
        train_transform = [
            A.RandomRotate90(p=0.9),
            A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            A.ToTensorV2()
        ]
    elif name == 'RandomRotate+BrightnessContrast':
        train_transform = [
            A.RandomRotate90(p=0.9),
            A.RandomBrightnessContrast(p=0.5),
            A.Resize(224, 224, interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            A.ToTensorV2()
        ]
    else:
        assert False, 'invalid name'

    return A.Compose(train_transform)

class GSI_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform = None):
        self.transform = transform
        self.dataset = glob.glob(path)
        self.datanum = len(self.dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        image = Image.open(self.dataset[idx])
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, 0

class GSI_DatasetWithMask(torch.utils.data.Dataset):
    mask_color_dict = {
        '17_Road'               : [255,0,0], 
        '23_Building_Normal'    : [255,0,255], 
        '29_Building_without_walls': [255,255,0],
        '32_Divider'            : [255,0,0]
    }

    def __init__(self, path, folder_name, transform = None):
        self.transform = transform
        self.dataset = []

        self.mask_color = GSI_DatasetWithMask.mask_color_dict[folder_name]

        for img_path in glob.glob( os.path.join(path, folder_name, 'org286', '*.png')) :

            msk_path = img_path.replace('org286', 'msk286')

            if os.path.exists(msk_path):
                self.dataset.append((img_path, msk_path, folder_name))
            else:
                self.dataset.append((img_path, None, folder_name))

        self.datanum = len(self.dataset)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):

        item = self.dataset[idx]

        img_data = np.array( Image.open(item[0]) )
        if item[1] is not None:
            msk_data = np.array( Image.open(item[1]) )
        else:
            msk_data = np.zeros((286, 286, 3))
        
        bk = np.zeros( (286, 286), dtype=np.uint8 )
        #bk[ np.where( (msk_data == [255,255,255]).all(axis=2) ) ] = 1
        #bk[ np.where( (msk_data == [255,0,0]).all(axis=2) ) ] = 1
        bk[ np.where( (msk_data == self.mask_color).all(axis=2) ) ] = 1
        # if np.max(bk) == 1.0 :
        #     print(f'{item[0]} contain road.')
        # else:
        #     print(f'{item[0]} not contain road.')

        label_onehot = torch.nn.functional.one_hot(torch.from_numpy(bk).long(), num_classes=2)
        label_onehot = label_onehot.to('cpu').detach().numpy().copy()
    
        if self.transform:
            augmented = self.transform(image=img_data, mask=label_onehot)
            image = augmented['image']
            mask = augmented['mask']

        mask = torch.permute( mask, (2,0,1) ).to( torch.float32 )

        return image, mask 
    
if __name__ == "__main__":
    transform_train = A.Compose([
        A.Resize(255, 255),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomBrightnessContrast(),
        A.Normalize(mean=(0,0,0), std=(1,1,1)),
        A.pytorch.ToTensorV2()
    ])

    dataset = GSI_DatasetWithMask( '/work/Dataset/GSI_Dataset/286/train', '17_Road', transform_train )

    for i in range(dataset.datanum):
        dataset.__getitem__(i)
