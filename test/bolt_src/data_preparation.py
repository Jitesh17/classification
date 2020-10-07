
# In[]:
import  pyjeasy.file_utils as f 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sys import exit as x
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import copy
from tqdm import tqdm
from PIL import Image
import glob

# get_ipython().run_line_magic('matplotlib', 'inline')

# In[]:
IMG_SIZE = 512
batch_size = 8
DATA_TYPES = ["train", "val", "test"]

# dataset_path = "/home/jitesh/3d/data/UE_training_results/bolt2/bolt_cropped"
# b_type_list = [dir for dir in os.listdir(dataset_path) if dir not in DATA_TYPES]
# print(b_type_list)


# # Required once start
# In[]:

def convert_4ch_to_3ch(dataset_path, split_ratio =[0.8, 0.1, 0.1]):
    b_type_list = [dir for dir in os.listdir(dataset_path) if dir not in DATA_TYPES]
    img_path_list = dict()
    for data_type in DATA_TYPES:
        img_path_list[data_type] = []
    import random
    for b_type in b_type_list:
        data = glob.glob(f'{dataset_path}/{b_type}/*.png')
    #     train_data.append(data[:int(len(data)*0.8)])
        random.shuffle(data)
        s1 = split_ratio[0]
        s2 = split_ratio[0] + split_ratio[1]
        assert 1 == split_ratio[0] + split_ratio[1] + split_ratio[2]
        img_path_list["train"] += data[:int(len(data)*s1)]
        img_path_list["val"] += data[int(len(data)*s1):int(len(data)*s2)]
        img_path_list["test"] += data[int(len(data)*s2):]
    print(f'len(train_data): {len(img_path_list["train"])}')
    print(f'len(val_data): {len(img_path_list["val"])}')
    print(f'len(test_data): {len(img_path_list["test"])}')


    # In[ ]:
    import  pyjeasy.file_utils as f 
    import cv2

    filename_list = dict()
    for data_type in DATA_TYPES:
        dirname_new = os.path.join(dataset_path, data_type)
        f.make_dir_if_not_exists(dirname_new)
        for file_path in tqdm(img_path_list[data_type]):
            file_path_split = file_path.split("/")
            dirname_old = file_path_split[-2].split("_")[0]
            filename_old = file_path_split[-1]
            # filename_new = dirname_old.replace("b", "") + "_" + filename_old
            filename_new = dirname_old + "_" + filename_old
            output_img_path = os.path.join(dirname_new, filename_new)
            f.delete_file_if_exists(output_img_path)
            
            # Converting 4 channel to 3 channel and then writing in different folder
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(output_img_path, img)
            # f.copy_file(src_path=file_path, dst_path=output_img_path, verbose=False)
        filename_list[data_type] = os.listdir(dirname_new)
    # train_files = os.listdir(TRAIN_IMG_DIR_PATH)
    # test_files = os.listdir(TEST_IMG_DIR_PATH)

# # Required once ends

# In[6]:
# filename_list = dict()
# for data_type in DATA_TYPES:
#     dirname_new = os.path.join(dataset_path, data_type)
#     filename_list[data_type] = os.listdir(dirname_new)

# In[6]:


class BoltDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform = None):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        # print(self.file_list[0])
        if self.mode == 'train':
            # if 'b00' in self.file_list[0]:
            # print(self.file_list[0])
            if 'b00' in self.file_list[0]:
                self.label = 0
            else:
                self.label = 1
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]
        
# data_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.ColorJitter(),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(128),
#     transforms.ToTensor()
# ])

def get_dataset(dataset_path: str, b_type_list: list, img_size: int):
    dataset = dict()
    
    filename_list = dict()
    for data_type in DATA_TYPES:
        dirname_new = os.path.join(dataset_path, data_type)
        filename_list[data_type] = os.listdir(dirname_new)
        
    for data_type in DATA_TYPES:
        dir_path = os.path.join(dataset_path, data_type)
        if data_type =="train":
            data_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ColorJitter(),
                transforms.RandomCrop(int(img_size*1)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])
            # catagory_data = dict()
            # for b_type in ['b00', 'b01']: #  b_type_list:
            #     # print(filename_list)
            #     cat_files = [tf for tf in filename_list[data_type] if b_type in tf]
            #     catagory_data[b_type] = BoltDataset(cat_files, dir_path, mode=data_type, transform = data_transform)
            # dataset[data_type] = ConcatDataset([c for c in catagory_data.values()])
        else:
            data_transform = transforms.Compose([
                transforms.Resize((img_size)),
                transforms.RandomCrop(int(img_size*1)),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])
            # dataset[data_type] = BoltDataset(filename_list[data_type], dir_path, mode=data_type, transform = data_transform)
        catagory_data = dict()
        # for b_type in ['b00', 'b01']: #  b_type_list:
        # for b_type in ['b10', 'b11']: #  b_type_list:
        for b_type in b_type_list:
            # print(filename_list)
            cat_files = [tf for tf in filename_list[data_type] if b_type in tf]
            catagory_data[b_type] = BoltDataset(cat_files, dir_path, mode=data_type, transform = data_transform)
        dataset[data_type] = ConcatDataset([c for c in catagory_data.values()])
        print(f'len({data_type}_data): {len(dataset[data_type])}')
    return dataset

# In[10]:
def mmmm():
    # batch_size = 2
    dataloader = DataLoader(dataset["train"], batch_size = batch_size, shuffle=True, num_workers=1)
    # dataloader = DataLoader(catdogs, batch_size = 32, shuffle=True, num_workers=4)
    print("len dataloader", len(dataloader))


    # In[30]:
    show_n_images = 40

    samples, labels = iter(dataloader).next()
    plt.figure(figsize=(16*2,24))
    grid_imgs = torchvision.utils.make_grid(samples[:show_n_images])
    np_grid_imgs = grid_imgs.numpy()
    # in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
    plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))


# In[]:
# In[]:
# In[]:

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))

# In[]:
convert_4ch_to_3ch("/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped")
# %%
