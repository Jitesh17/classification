#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
print('torch.cuda.is_available():   ', torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise EnvironmentError
print(torch.cuda.get_device_name(0))
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


# In[5]:

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
import tqdm
from PIL import Image
import glob

# get_ipython().run_line_magic('matplotlib', 'inline')


# # Data preparation

# In[11]:
IMG_SIZE = 128
batch_size = 8
data_types = ["train", "val", "test"]

dataset_path = "/home/jitesh/3d/data/UE_training_results/bolt"
dataset_path = "/home/jitesh/3d/data/UE_training_results/bolt2/bolt_cropped"
b_type_list = [dir for dir in os.listdir(dataset_path) if dir not in data_types]
print(b_type_list)

# In[11]:
# data_types = ["train", "val", "test"]
img_path_list = dict()
for data_type in data_types:
    img_path_list[data_type] = []
train_data = []
val_data = []
test_data = []
import random
for b_type in b_type_list:
    data = glob.glob(f'{dataset_path}/{b_type}/*.png')
#     train_data.append(data[:int(len(data)*0.8)])
    random.shuffle(data)
    img_path_list["train"] += data[:int(len(data)*0.8)]
    img_path_list["val"] += data[int(len(data)*0.8):int(len(data)*0.9)]
    img_path_list["test"] += data[int(len(data)*0.9):]
print(f'len(train_data): {len(img_path_list["train"])}')
print(f'len(val_data): {len(img_path_list["val"])}')
print(f'len(test_data): {len(img_path_list["test"])}')

# TRAIN_IMG_DIR_PATH = f"{dataset_path}/train"
# TEST_IMG_DIR_PATH = f"{dataset_path}/test1"
# train_files = os.listdir(TRAIN_IMG_DIR_PATH)
# test_files = os.listdir(TEST_IMG_DIR_PATH)


# # Required once start

# # In[ ]:
# import  pyjeasy.file_utils as f 
# import cv2

# filename_list = dict()
# for data_type in data_types:
#     dirname_new = os.path.join(dataset_path, data_type)
#     f.make_dir_if_not_exists(dirname_new)
#     for file_path in img_path_list[data_type]:
#         file_path_split = file_path.split("/")
#         dirname_old = file_path_split[-2].split("_")[0]
#         filename_old = file_path_split[-1]
#         # filename_new = dirname_old.replace("b", "") + "_" + filename_old
#         filename_new = dirname_old + "_" + filename_old
#         output_img_path = os.path.join(dirname_new, filename_new)
#         f.delete_file_if_exists(output_img_path)
        
#         # Converting 4 channel to 3 channel and then writing in different folder
#         img = cv2.imread(file_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#         cv2.imwrite(output_img_path, img)
#         # f.copy_file(src_path=file_path, dst_path=output_img_path, verbose=False)
#     filename_list[data_type] = os.listdir(dirname_new)
# # train_files = os.listdir(TRAIN_IMG_DIR_PATH)
# # test_files = os.listdir(TEST_IMG_DIR_PATH)

# # Required once ends

# In[6]:
filename_list = dict()
for data_type in data_types:
    dirname_new = os.path.join(dataset_path, data_type)
    filename_list[data_type] = os.listdir(dirname_new)

# In[6]:


class BoltDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', transform = None):
        self.file_list = file_list
        self.dir = dir
        self.mode= mode
        self.transform = transform
        if self.mode == 'train':
            # if 'b00' in self.file_list[0]:
            if 'b00' in self.file_list[0]:
                self.label = 0
            elif 'b01' in self.file_list[0]:
                self.label = 1
            elif 'b10' in self.file_list[0]:
                self.label = 2
            elif 'b11' in self.file_list[0]:
                self.label = 3
            # else:
            #     self.label = 1
            
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

dataset = dict()
for data_type in data_types:
    dir_path = os.path.join(dataset_path, data_type)
    if data_type =="train":
        data_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ColorJitter(),
            transforms.RandomCrop(int(IMG_SIZE*1)),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(IMG_SIZE),
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
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor()
        ])
        # dataset[data_type] = BoltDataset(filename_list[data_type], dir_path, mode=data_type, transform = data_transform)
    catagory_data = dict()
    # for b_type in ['b00', 'b01']: #  b_type_list:
    for b_type in ['b00', 'b01', 'b10', 'b11']: #  b_type_list:
    # for b_type in b_type_list:
        # print(filename_list)
        cat_files = [tf for tf in filename_list[data_type] if b_type in tf]
        catagory_data[b_type] = BoltDataset(cat_files, dir_path, mode=data_type, transform = data_transform)
    dataset[data_type] = ConcatDataset([c for c in catagory_data.values()])
    print(f'len({data_type}_data): {len(dataset[data_type])}')

# In[6]:

# # In[6]:


# class CatDogDataset(Dataset):
#     def __init__(self, file_list, dir, mode='train', transform = None):
#         self.file_list = file_list
#         self.dir = dir
#         self.mode= mode
#         self.transform = transform
#         if self.mode == 'train':
#             if 'dog' in self.file_list[0]:
#                 self.label = 1
#             else:
#                 self.label = 0
            
#     def __len__(self):
#         return len(self.file_list)
    
#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.dir, self.file_list[idx]))
#         if self.transform:
#             img = self.transform(img)
#         if self.mode == 'train':
#             img = img.numpy()
#             return img.astype('float32'), self.label
#         else:
#             img = img.numpy()
#             return img.astype('float32'), self.file_list[idx]
        
# data_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.ColorJitter(),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(128),
#     transforms.ToTensor()
# ])

# cat_files = [tf for tf in train_files if 'cat' in tf]
# dog_files = [tf for tf in train_files if 'dog' in tf]

# cats = CatDogDataset(cat_files, TRAIN_IMG_DIR_PATH, transform = data_transform)
# dogs = CatDogDataset(dog_files, TRAIN_IMG_DIR_PATH, transform = data_transform)

# catdogs = ConcatDataset([cats, dogs])


# # In[22]:


# len(catdogs)


# # In[10]:
# dataset["train"].shape

# In[10]:

# batch_size = 2
dataloader = DataLoader(dataset["train"], batch_size = batch_size, shuffle=True, num_workers=2)
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


# In[31]:


device = 'cuda'
# model = torchvision.models.densenet121(pretrained=True)
# torchvision.models

# In[32]:

model = torchvision.models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 500),
    torch.nn.Linear(500, 4)
)

model = model.to(device)
loss_criteria = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*batch_size, 50*batch_size], gamma=0.5)


# In[33]:


epochs = 3
epochs = 5
itr = 1
p_itr = batch_size #200
model.train()
total_loss = 0
loss_list = []
acc_list = []
val_total_loss = 0
val_loss_list = []
val_acc_list = []
# print("len dataloader", len(dataloader))
for epoch in range(epochs):
    for samples, labels in dataloader:
        # print(labels)
        # print(labels.shape)
        samples, labels = samples.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(samples)
        loss = loss_criteria(output, labels)
        loss.backward()
        total_loss += loss.item()
        ###
        val_output = model(val_sample)
        val_loss = loss_criteria(output, torch.ones(8, dtype=torch.long).to(device)*3)
        val_total_loss += val_loss.item()
        ###
        optimizer.step()
        # scheduler.step()
        if itr%p_itr == 0:
            pred = torch.argmax(output, dim=1)
            correct = pred.eq(labels)
            acc = torch.mean(correct.float())
            loss_list.append(total_loss/p_itr)
            acc_list.append(acc)
            ###
            val_pred = torch.argmax(val_output, dim=1)
            val_correct = pred.eq(torch.ones(8, dtype=torch.long).to(device)*3)
            val_acc = torch.mean(val_correct.float())
            val_loss_list.append(val_total_loss/p_itr)
            val_acc_list.append(val_acc)
            ###
            
            if itr%p_itr == 0:
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
            # loss_list.append(total_loss/p_itr)
            # acc_list.append(acc)
            total_loss = 0
            val_total_loss = 0

        itr += 1
    plt.plot(loss_list, label='loss')
    plt.plot(acc_list, label='accuracy')
    plt.plot(val_acc_list, label='val_accuracy')
    plt.legend()
    plt.title('training loss and accuracy')
    plt.show()
    ###
    plt.plot(val_loss_list, label='loss')
    plt.plot(val_acc_list, label='accuracy')
    plt.legend()
    plt.title('training loss and accuracy')
    plt.show()
print("Total iterations: ", itr-1)

plt.plot(loss_list, label='loss')
plt.plot(acc_list, label='accuracy')
plt.legend()
plt.title('training loss and accuracy')
plt.show()


# In[35]:


# filename_pth = 'ckpt_densenet121_catdog.pth'
# filename_pth = f'ckpt_densenet121_mark_exist_{IMG_SIZE}.pth'
filename_pth = f'ckpt_densenet121_mark_correct_{IMG_SIZE}_4class_2.pth'
torch.save(model.state_dict(), filename_pth)

# In[36]:
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor()
])

# testset = CatDogDataset(test_files, TEST_IMG_DIR_PATH, mode='test', transform = test_transform)
# testloader = DataLoader(testset, batch_size = 32, shuffle=False, num_workers=4)
testloader = DataLoader(dataset["test"], batch_size = batch_size, shuffle=False, num_workers=2)


# In[36]:


model.eval()
fn_list = []
pred_list = []
for x, fn in testloader:
    with torch.no_grad():
        x = x.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        fn_list += [n[:-4] for n in fn]
        pred_list += [p.item() for p in pred]

submission = pd.DataFrame({"id":fn_list, "label":pred_list})
submission.to_csv(f'preds_densenet121_dir_{IMG_SIZE}__4class_2.csv', 
                #   index=False
                  )
# In[37]:

testloader = DataLoader(dataset["test"], batch_size = 12, shuffle=True, num_workers=2)

samples, _ = iter(testloader).next()
samples = samples.to(device)
fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((1.0, 0.47, 0.42))
fig.tight_layout()
output = model(samples[:24])
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
# ad = {0:'cat', 1:'dog'}
# ad = {0:'no mark', 1:'mark'}
ad = {0:'Incorrect', 1:'Correct'}
# for num, sample in enumerate(samples[:24]):
for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    # plt.axis('off')
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))
# ax = plt.gca()
plt.savefig('inference_mark_direction__4class.png')

# In[ ]:
# imsize = 128
# loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

# def image_loader(image_name):
#     """load image, returns cuda tensor"""
#     image = Image.open(image_name)
#     image = test_transform(image).float()
#     image = Variable(image, requires_grad=True)
#     image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#     return image.cuda()  #assumes that you're using GPU
img_path = "/home/jitesh/3d/data/UE_training_results/bolt/b11/000001.png"
img_path = "/home/jitesh/3d/data/UE_training_results/bolt/val/b10_000496.png"
img_path = "/home/jitesh/3d/data/UE_training_results/bolt2/bolt_cropped/test/b01_000810.png"
dir_path = os.path.abspath(f"{img_path}/..")
test_files = [img_path.split('/')[-1]]
# image = image_loader(img_path)
test_data = BoltDataset(test_files, dir_path, mode="test", transform = data_transform)
testloader = DataLoader(test_data, batch_size = 12, shuffle=False, num_workers=1)
# your_trained_net(image)



model.eval()
fn_list = []
pred_list = []
for x, fn in testloader:
    with torch.no_grad():
        x = x.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)

label_int = [p.item() for p in pred]
print(label_int)
# %%

MODEL_WEIGHT_PATH = "/home/jitesh/prj/classification/test/bolt/ckpt_densenet121_mark_correct_128_4class_2.pth"
model.load_state_dict(torch.load(MODEL_WEIGHT_PATH))

# In[37]:
data_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor()
        ])
test_dir_path = "/home/jitesh/sekisui/bolt/cropped_hexagon_bolts"
test_dir_path = "/home/jitesh/sekisui/bolt/cropped_hexagon_bolts/b11"
# test_dir_path = "/home/jitesh/sekisui/bolt/cropped_hexagon_bolts/b10"
test_list = [file for file in os.listdir(test_dir_path) if os.path.isfile(os.path.join(test_dir_path, file))]
test_list = sorted(test_list)
# print(test_list)
test_data = BoltDataset(test_list, test_dir_path, mode="test", transform = data_transform)
testloader = DataLoader(test_data, batch_size = 12, shuffle=False, num_workers=2)

model.eval()
fn_list = []
pred_list = []
for x, fn in testloader:
    with torch.no_grad():
        x = x.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        fn_list += [n[:-4] for n in fn]
        pred_list += [p.item() for p in pred]

submission = pd.DataFrame({"id":fn_list, "label":pred_list})
submission.to_csv(f'preds_densenet121_dir_{IMG_SIZE}_test_4classes_2.csv', 
                #   index=False
                  )
samples, _ = iter(testloader).next()
samples = samples.to(device)
fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((1.0, 0.47, 0.42))
fig.tight_layout()
output = model(samples[:24])
val_sample = samples[:24]
pred = torch.argmax(output, dim=1)
pred = [p.item() for p in pred]
# ad = {0:'cat', 1:'dog'}
# ad = {0:'no mark', 1:'mark'}
ad = {0:'0', 1:'1', 2:'3', 3:'3'}
# for num, sample in enumerate(samples[:24]):
for num, sample in enumerate(samples[:24]):
    plt.subplot(4,6,num+1)
    plt.title(ad[pred[num]])
    # plt.axis('off')
    # val_sample = sample
    sample = sample.cpu().numpy()
    plt.imshow(np.transpose(sample, (1,2,0)))
# ax = plt.gca()
plt.savefig('inference_mark_direction_test_4classes_2.png')
# %%
