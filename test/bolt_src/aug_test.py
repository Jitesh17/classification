
# In[]:
import os, sys
import  pyjeasy.file_utils as f 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from RandAugment import RandAugment
import albumentations as A
from PIL import Image
from pyjeasy.image_utils.output import show_image

path = "/home/jitesh/3d/data/UE_training_results/bolt2/bolt_cropped/b01/000001.png"

cv2_img = cv2.imread(path)
pil_img = Image.open(path)

# show_image(cv2_img, 800)
# In[]:
IMG_SIZE = 128
# aug_cv2_img = albu.GaussNoise(always_apply=True, p=1)(image=cv2_img)['image']
def get_augmentation(img):
    train_transform = [
        albu.Resize(height=IMG_SIZE, width= IMG_SIZE, p=1),
        # albu.GaussNoise(p=1),
        # albu.Blur(blur_limit=3, p=1),
        albu.GaussianBlur(blur_limit=3, p=1),
        
    ]
    transforms = albu.Compose(train_transform)  # <- Compose
    return transforms(image=img)['image'], transforms

aug_cv2_img, data_transform = get_augmentation(cv2_img)

plt.figure(figsize=(8, 5))
plt.imshow(aug_cv2_img)
# show_image(aug_cv2_img, 800)
# cv2.waitKey()
# %%
class BoltDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform = None, test_label: int=1):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        self.test_label = test_label
        if self.mode == 'train':
            # print(self.file_list)
            # if 'b00' in self.file_list[0]:
            if 'b10' in self.file_list[0]:
                self.label = 0
            else:
                self.label = 1
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        print('hi', idx)
        idx = idx % len(self.file_list)
        print('hi%V ', idx)
        img = Image.open(self.file_list[idx])
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        elif self.mode == 'test':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]
data = BoltDataset(file_list=[path]*20, dir="dir_path", mode="train", transform = data_transform)
dataloader = DataLoader(data, batch_size = 1, shuffle=True, num_workers=1)
# %%
samples, labels = iter(dataloader).next()
# %%
plt.figure(figsize=(16*2,24))
grid_imgs = torchvision.utils.make_grid(samples[:1])
np_grid_imgs = grid_imgs.numpy()
plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
# %%
a=list((path,))
a[0]

# %%
