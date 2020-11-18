#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations
# from _typeshed import NoneType

import copy
import glob
import os
from datetime import datetime
from shutil import Error
from sys import exit as x
from typing import List, Union
from zipfile import error

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import printj
import pyjeasy.file_utils as f
import torch
import torch.nn as nn
import torchvision
import tqdm
# import albumentations as albu
from imgaug import augmenters as iaa
from PIL import Image
# from RandAugment import RandAugment
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.transforms import RandomGrayscale
from jaitool.training import save_ckp, load_ckp
# writer = SummaryWriter()
# get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def check_cuda():
    print('torch.cuda.is_available():   ', torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise EnvironmentError
    print(torch.cuda.get_device_name(0))
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')


class BoltDataset(Dataset):
    def __init__(self, file_list, dir, mode='train', aug=None, 
                 transform=None, test_label: int = 1, 
                 b_type_list: List[str] = ['b10', 'b11'], 
                 img_size: int=256):
        # super().__init__()
        self.file_list = file_list
        self.dir = dir
        self.mode = mode
        # self.transform = transform
        self.test_label = test_label
        self.b_type_list = b_type_list
        self.img_size=img_size
        if self.mode == 'train' or self.mode == 'val':
            # print(self.file_list)
            # if 'b00' in self.file_list[0]:
            if b_type_list[0] in self.file_list[0]:
                self.label = 0
            else:
                self.label = 1
        if aug is None:
            self.aug = BoltClassifier(img_size=img_size).get_augmentation()
        self.val_aug_seq = A.Compose([
            A.Resize(self.img_size, self.img_size),
            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.file_list[idx])
        image = Image.open(img_path)
        big_side = max(image.size)
        # small_side = min(image.size)
        # printj.red(list(image.size)[0])
        # print(big_side)
        new_im = Image.new("RGB", (big_side, big_side))
        new_im.paste(image)
        image = new_im
        # x()
        if self.mode == 'train':
            image = self.aug(image=np.array(image))['image']
            # image = self.val_aug_seq(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return torch.tensor(image, dtype=torch.float), self.label
        elif self.mode == 'val':
            image = self.val_aug_seq(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return torch.tensor(image, dtype=torch.float), self.label
        else:
            image = self.val_aug_seq(image=np.array(image))['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return torch.tensor(image, dtype=torch.float), self.file_list[idx]


class BoltClassifier:
    def __init__(self, device: str = 'cuda', img_size: int = 256, batch_size: int = 8, data_types: List[str] = ["train", "val", "test"],
                dataset_path: str = "/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped",
                b_type_list: List[str] = ['b10', 'b11'], num_workers: int = 2):
        self.device = device
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_types = data_types
        self.dataset_path = dataset_path
        self.b_type_list = b_type_list
        self.num_workers = num_workers
        self.scheduler = None
        self.model = None
        # self.set_model()
        # self.dataloader = self.get_dataloader()

    def get_val(self, model, model_path, test_dir_path, no_of_samples: int = 24, test_label: int = 1, save_csv_path: str = None):
        model.load_state_dict(torch.load(model_path))
        # "/home/jitesh/prj/classification/test/bolt/ckpt_densenet121_mark_correct_128_s3_5.pth"))
        data_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        test_list = [file for file in os.listdir(test_dir_path) if os.path.isfile(
            os.path.join(test_dir_path, file)) and "b1" in file]
        test_list = sorted(test_list)
        # print(test_list)
        test_data = BoltDataset(test_list, test_dir_path, mode="test",
                                transform=data_transform, test_label=test_label)
        testloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        model.eval()
        fn_list = []
        pred_list = []
        for x, fn in testloader:
            with torch.no_grad():
                x = x.to(self.device)
                output = model(x)
                pred = torch.argmax(output, dim=1)
                fn_list += [n[:-4] for n in fn]
                pred_list += [p.item() for p in pred]

        submission = pd.DataFrame({"id": fn_list, "label": pred_list})
        if save_csv_path is None:
            save_csv_path = f'preds_densenet121_dir_{self.img_size}_test_.csv'
        submission.to_csv(save_csv_path,
                          #   index=False
                          )
        samples, _ = iter(testloader).next()
        samples = samples.to(self.device)
        val_sample = samples[:no_of_samples]
        return val_sample

    def predict(self, model=None, model_path=None, test_dir_path=None, 
                no_of_samples: int = 24, test_label: int = 1, 
                save_csv_path: str = None, write_images=None):
        if write_images:
            f.make_dir_if_not_exists(write_images)
            f.delete_all_files_in_dir(write_images)
            f.make_dir_if_not_exists(f'{write_images}/{self.b_type_list[0]}')
            f.make_dir_if_not_exists(f'{write_images}/{self.b_type_list[1]}')
        if model is None:
            model = self.model
        try:
            model.load_state_dict(torch.load(model_path)['state_dict'])
        except KeyError:
            model.load_state_dict(torch.load(model_path))
        # "/home/jitesh/prj/classification/test/bolt/ckpt_densenet121_mark_correct_128_s3_5.pth"))
        data_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.ToTensor(),
            ]) 
        test_list = [file for file in os.listdir(test_dir_path) if os.path.isfile(
            os.path.join(test_dir_path, file))] # and "b1" in file]
        test_list = sorted(test_list)
        # print(test_list)
        test_data = BoltDataset(test_list, test_dir_path, mode="test",
                                transform=data_transform, test_label=test_label)
        testloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        model.eval()
        fn_list = []
        pred_list = []
        for x, fn in testloader:
            with torch.no_grad():
                x = x.to(self.device)
                output = model(x)
                pred = torch.argmax(output, dim=1)
                fn_list += [n[:-4] for n in fn]
                pred_list += [p.item() for p in pred]
                if write_images:
                    for filename, prediction in zip(fn, pred):
                        if prediction==0:
                            f.copy_file(src_path=f'{test_dir_path}/{filename}',
                                        dst_path=f'{write_images}/{self.b_type_list[0]}/{filename}')
                        elif prediction==1:
                            f.copy_file(src_path=f'{test_dir_path}/{filename}',
                                        dst_path=f'{write_images}/{self.b_type_list[1]}/{filename}')

        submission = pd.DataFrame({"id": fn_list, "label": pred_list})
        if save_csv_path is None:
            save_csv_path = f'preds_densenet121_dir_{self.img_size}_test_.csv'
        elif '.csv' not in save_csv_path:
            save_csv_path = save_csv_path + '/1.csv'
        submission.to_csv(save_csv_path,
                          #   index=False
                          )
            
            
            
        # samples, _ = iter(testloader).next()
        # samples = samples.to(self.device)
        # val_sample = samples[:no_of_samples]
        # return val_sample

    @staticmethod
    def convert_img_shape_aug_to_normal(img):
        # unnormalize
        npimg = (img / 2 + 0.5)*255
        # npimg = np.clip(npimg, 0, 255)
        # print((npimg))
        # print((npimg.shape))
        # x()
        # npimg = npimg.astype(int)
        return Image.fromarray(npimg.astype('uint8'), 'RGB')

    @staticmethod
    def convert_img_shape_tensor_to_normal(img):
        # unnormalize
        img = img / 2 + 0.5
        npimg = img.numpy()
        npimg = np.clip(npimg, 0., 1.)
        
        return np.transpose(npimg, (1, 2, 0))

    @staticmethod
    def tensore_to_np(img):
        # unnormalize
        img = img / 2 + 0.5
        npimg = img.numpy()
        npimg = np.clip(npimg, 0., 1.)
        return np.transpose(npimg, (1, 2, 0))

    def show_img(self, img):
        plt.figure(figsize=(18, 15))
        plt.imshow(self.tensore_to_np(img))
        plt.show()

    def split_train_test_0(self, split_ratio=[0.8, 0.1, 0.1], verbose: bool = True):
        import random
        if self.b_type_list is None:
            self.b_type_list = [dir for dir in os.listdir(
                self.dataset_path) if dir not in self.data_types]

        img_path_list = dict()
        for data_type in self.data_types:
            img_path_list[data_type] = []

        for b_type in self.b_type_list:
            data = glob.glob(f'{self.dataset_path}/{b_type}/*.png')
            random.shuffle(data)
            s1 = split_ratio[0]
            s2 = split_ratio[0] + split_ratio[1]
            assert 1 == split_ratio[0] + split_ratio[1] + split_ratio[2]
            img_path_list["train"] += data[:int(len(data)*s1)]
            img_path_list["val"] += data[int(len(data)*s1):int(len(data)*s2)]
            img_path_list["test"] += data[int(len(data)*s2):]
        if verbose:
            print(f'len(train_data): {len(img_path_list["train"])}')
            print(f'len(val_data): {len(img_path_list["val"])}')
            print(f'len(test_data): {len(img_path_list["test"])}')
        return img_path_list

    def split_train_test(self, split_ratio=[0.8, 0.1, 0.1], verbose: bool = True):
        import random
        # if self.b_type_list is None:
        #     self.b_type_list = [dir for dir in os.listdir(
        #         self.dataset_path) if dir not in self.data_types]

        img_path_list = dict()
        for data_type in ["train", "val", "test"]:
            img_path_list[data_type] = []

        # for b_type in self.b_type_list:
        # data = glob.glob(f'{self.dataset_path}/train/*')
        data = os.listdir(f'{self.dataset_path}/train')
        # print(data)
        random.shuffle(data)
        s1 = split_ratio[0]
        s2 = split_ratio[0] + split_ratio[1]
        assert 1 == split_ratio[0] + split_ratio[1] + split_ratio[2]
        img_path_list["train"] += data[:int(len(data)*s1)]
        img_path_list["val"] += data[int(len(data)*s1):int(len(data)*s2)]
        img_path_list["test"] += data[int(len(data)*s2):]
        if verbose:
            print(f'len(train_data): {len(img_path_list["train"])}')
            print(f'len(val_data): {len(img_path_list["val"])}')
            print(f'len(test_data): {len(img_path_list["test"])}')
            print(f'len(test_data): {(img_path_list["val"])}')
        self.filename_list = img_path_list
        return img_path_list

    def get_image_list(self):
        filename_list = dict()
        for data_type in self.data_types:
            dirname_new = os.path.join(self.dataset_path, data_type)
            filename_list[data_type] = os.listdir(dirname_new)
            # printj.red(filename_list[data_type])
            # printj.yellow(dirname_new)
        self.filename_list = filename_list
        return filename_list

    # aug_iaa = iaa.Sequential([
    #                 iaa.flip.Fliplr(p=0.5),
    #                 iaa.flip.Flipud(p=0.5),
    #                 iaa.GaussianBlur(sigma=(0.0, 0.1)),
    #                 iaa.MultiplyBrightness(mul=(0.65, 1.35)),
    #             ])
    
    def get_augmentation(self, save_path=None, load_path=None):
        if load_path:
            return A.load(load_path)
        else:
            aug_seq1 = A.OneOf([
                A.Rotate(limit=(-90, 90), p=1.0),
                A.Flip(p=1.0),
            ], p=1.0)
            aug_seq2 = A.OneOf([
                # A.ChannelDropout(always_apply=False, p=1.0, channel_drop_range=(1, 1), fill_value=0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=1.0),
                A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(
                    -0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True)
            ], p=1.0)
            aug_seq3 = A.OneOf([
                A.GaussNoise(always_apply=False, p=1.0, var_limit=(10, 100)),
                A.ISONoise(always_apply=False, p=1.0, intensity=(
                    0.1, 1.0), color_shift=(0.01, 0.3)),
                A.MultiplicativeNoise(always_apply=False, p=1.0, multiplier=(
                    0.8, 1.6), per_channel=True, elementwise=True),
            ], p=1.0)
            aug_seq4 = A.OneOf([
                A.Equalize(always_apply=False, p=1.0,
                           mode='pil', by_channels=True),
                A.InvertImg(always_apply=False, p=1.0),
                A.MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 7)),
                A.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.3, 0.3), 
                                    shift_limit=(-0.05, 0.05), interpolation=0, 
                                    border_mode=0, value=(0, 0, 0), mask_value=None),
                A.RandomFog(always_apply=False, p=1.0, fog_coef_lower=0.1,
                            fog_coef_upper=0.45, alpha_coef=0.5)
            ], p=1.0)
            aug_seq = A.Compose([
                A.Resize(self.img_size, self.img_size),
                aug_seq1,
                aug_seq2,
                aug_seq3,
                aug_seq4,
                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            # aug_path = '/home/jitesh/prj/classification/test/bolt/aug/aug_seq.json'
            if save_path:
                A.save(aug_seq, save_path)
            # loaded_transform = A.load(aug_path)
            return aug_seq

    def view_data(self):
        cat_files = [tf for tf in self.filename_list["train"]
                     if "10" in tf]  # if b_type in tf]
        dir_path = os.path.join(self.dataset_path, "train")
        alb_dataset = BoltDataset(cat_files, dir_path)
        alb_dataloader = DataLoader(
            dataset=alb_dataset, batch_size=16, shuffle=True)
        data = iter(alb_dataloader)
        images, _ = data.next()
        # show images
        self.show_img(torchvision.utils.make_grid(images))

    def get_dataset(self):
        dataset = dict()
        
        for data_type in self.data_types:
            dir_path = os.path.join(self.dataset_path, "train")
            # dir_path = os.path.join(self.dataset_path, data_type)
            catagory_data = dict()
            # for b_type in ['b00', 'b01']: #  b_type_list:
            # for b_type in ['b10', 'b11']:  # b_type_list:
            for b_type in self.b_type_list:
                # print(filename_list[data_type])
                cat_files = [
                    tf for tf in self.filename_list[data_type] if b_type in tf]

                # print(cat_files)
                # , transform = albu_transforms)#data_transform)
                catagory_data[b_type] = BoltDataset(
                    cat_files, dir_path, mode=data_type, 
                    b_type_list=self.b_type_list, img_size=self.img_size)
            dataset[data_type] = ConcatDataset(
                [c for c in catagory_data.values()])
            print(f'len({data_type}_data): {len(dataset[data_type])}')
        return dataset

    def get_dataloader(self, data=None):
        if data is None:
            dataset = self.get_dataset()
            data = dataset["train"]
        dataloader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print("len dataloader", len(dataloader))
        self.dataloader = dataloader
        return dataloader

    def view_data2(self, dataloader=None, show_n_images: int = 24):
        if dataloader is None:
            dataloader = self.get_dataloader()
        # images, labels = dataloader.next()
        # self.show_img(torchvision.utils.make_grid(images))
        # # show_n_images = 40

        samples, labels = iter(dataloader).next()
        # plt.figure(figsize=(16*2, 24))
        grid_imgs = torchvision.utils.make_grid(samples[:show_n_images])
        # np_grid_imgs = grid_imgs.numpy()
        # in tensor, image is (batch, width, height), so you have to transpose it to (width, height, batch) in numpy to show it.
        # plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
        self.show_img(grid_imgs)
        # plt.savefig("augr.png")

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def set_model(self, learning_rate: float = 0.001, scheduler_on :bool=False, scheduler_steps: list=[500,1000,1500], dataloader=None):
        if dataloader is None:
            dataloader = self.get_dataloader()
        model = torchvision.models.densenet121(pretrained=False)
        # model = torchvision.models.googlenet(pretrained=False)
        self.num_ftrs = num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 500),
            torch.nn.Linear(500, 2)
            # torch.nn.Linear(500, 1)
        )
        # model.features.
        self.model = model.to(self.device)
        self.loss_criteria = nn.CrossEntropyLoss()
        # loss_criteria = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=True)
        if scheduler_on:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=scheduler_steps, gamma=0.5)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[ 100*self.batch_size], gamma=0.5)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     self.optimizer, milestones=[200], gamma=0.5)
            # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(dataloader), eta_min=learning_rate)
        # return model,
    # # In[33]:
    # val_sample = get_val(model=model,
    #                      model_path="/home/jitesh/prj/classification/test/bolt/ckpt_densenet121_mark_correct_128_s3_2.pth",
    #                      test_dir_path="/home/jitesh/sekisui/bolt/cropped_hexagon_bolts/b11",
    #                      )
    # # In[33]:

    def train(self, epochs: int = 5,  val_check_period = None, 
              val_sample=None, 
              dataloader_train=None, dataloader_val=None, 
              show_tensorboard=True, tensorboard_log_dir_path=None, 
              show_pyplot=False, weight_dir_path="weights", writer=None,
              iteration_limit=np.inf):
        data_available = False
        if show_tensorboard:
            if tensorboard_log_dir_path:
                writer = SummaryWriter(log_dir=tensorboard_log_dir_path)
            else:
                writer = SummaryWriter()
        else:
            writer = None
        dataset = self.get_dataset()
        if dataloader_train is None:
            dataloader_train = self.get_dataloader(dataset["train"])
        if dataloader_val is None:
            dataloader_val = DataLoader(dataset["val"], 
                                        batch_size=self.batch_size,
                                        # batch_size=len(dataset["val"]), 
                                        shuffle=False, num_workers=self.num_workers)
            # print(len(dataloader_val))
        for val_samples, val_labels in dataloader_val:
            val_samples = val_samples.to(self.device)
            val_labels = val_labels.to(self.device)
        # epochs = 3
        # epochs = 10
        itr = 1
        p_itr = self.batch_size  # 200
        self.model.train()
        total_loss = 0
        loss_list = []
        acc_list = []
        val_total_loss = 0
        val_loss_list = []
        val_acc_list = []
        
        # if val_check_period is None:
        #     val_check_period = len(dataloader_train)
        # print("len dataloader", len(dataloader))
        # dataloader_train=self.get_dataloader()
        
        for epoch in range(epochs):
            if itr > iteration_limit:
                break
            for batch_idx, (samples, labels) in enumerate(dataloader_train):
                # all_val_labels = []
                # all_val_output = []
                # print(labels)
                # print(labels.shape)
                samples, labels = samples.to(
                    self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(samples)
                loss = self.loss_criteria(output, labels)
                loss.backward()
                total_loss += loss.item()
                ###
                
                if val_sample:
                    val_output = self.model(val_sample)
                    val_loss = self.loss_criteria(val_output, torch.ones(
                        8, dtype=torch.long).to(self.device))
                    val_total_loss += val_loss.item()
                elif val_check_period:
                    if itr % val_check_period == 0:
                        val_output = self.model(val_samples)
                        val_loss = self.loss_criteria(val_output, val_labels)
                        val_total_loss += val_loss.item()
                    # all_val_labels += val_labels.to(self.device)
                    # all_val_output += val_output
                else:
                    val_output = output
                ###
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                if itr % p_itr == 0:
                    pred = torch.argmax(output, dim=1)
                    correct = pred.eq(labels)
                    acc = torch.mean(correct.float())
                    loss_list.append(total_loss/p_itr)
                    acc_list.append(acc)
                    ###
                    val_acc = 0
                    if val_sample:
                        val_pred = torch.argmax(val_output, dim=1)
                        val_correct = val_pred.eq(torch.ones(
                            8, dtype=torch.long).to(self.device))
                        val_acc = torch.mean(val_correct.float())
                        val_loss_list.append(val_total_loss/p_itr)
                        val_acc_list.append(val_acc)

                        print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}, Val Accuracy: {:.3f}'.format(
                            epoch+1, epochs, itr, total_loss/p_itr, acc, val_acc))
                    elif val_check_period:
                        if itr % val_check_period == 0:
                            val_pred = torch.argmax(val_output, dim=1)
                            val_correct = val_pred.eq(val_labels)
                            val_acc = torch.mean(val_correct.float())
                            val_loss_list.append(val_total_loss/val_check_period)
                            val_acc_list.append(val_acc)
                        
                            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}, Val Accuracy: {:.3f}'.format(
                                epoch+1, epochs, itr, total_loss/p_itr, acc, val_acc))
                     
                        else:
                            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(
                                epoch+1, epochs, itr, total_loss/p_itr, acc))
                    else:
                        print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(
                            epoch+1, epochs, itr, total_loss/p_itr, acc))
                    # loss_list.append(total_loss/p_itr)
                    # acc_list.append(acc)
                    if show_tensorboard:
                        # samples = samples / 2 + 0.5
                        # img_grid = torchvision.utils.make_grid(samples)
                        # writer.add_image('training data', img_grid)
                        # val_img_grid = torchvision.utils.make_grid(val_samples)
                        # writer.add_image('val data', val_img_grid)
                        # writer.add_graph(self.model, samples)
                        
                        # log embeddings
                        # features = samples.view(-1, 3*self.img_size*self.img_size)
                        data = samples
                        data_labels = labels
                        # data = val_samples
                        # data_labels = val_labels
                        # features = data.reshape(data.shape[0], -1)
                        # class_labels = [self.b_type_list[label] for label in data_labels]
                        # writer.add_embedding(features,
                        #     metadata=class_labels,
                        #     label_img=data,
                        #     global_step=batch_idx
                        #     # label_img=samples.unsqueeze(1),
                        #     )
                        # features = val_samples.reshape(val_samples.shape[0], -1)
                        # class_labels = [self.b_type_list[label] for label in val_labels]
                        # writer.add_embedding(features,
                        #     metadata=class_labels,
                        #     label_img=val_samples,
                        #     global_step=batch_idx
                        #     # metadata=self.b_type_list,
                        #     # label_img=samples.unsqueeze(1),
                        #     )

                        writer.add_scalar('* Loss/train', total_loss/p_itr, itr)
                        writer.add_scalar('* Accuracy/train', acc, itr)
                        writer.add_scalar('Learning Rate', self.get_lr(), itr)
                    
                    if show_tensorboard:
                        if val_sample or val_check_period:
                            if itr % val_check_period == 0:
                                # print(f"{val_total_loss}/{val_check_period} = {val_total_loss/val_check_period}")
                                writer.add_scalar('* Loss/val', val_total_loss/val_check_period, itr)
                                writer.add_scalar('* Accuracy/val', val_acc, itr)
                                
                            if itr > (epochs)*len(dataloader_train) -p_itr - 2:
                                writer.add_hparams({
                                    'Learning rate': self.get_lr(),
                                    'Batch Size': self.batch_size,
                                    'Image Size': self.img_size,
                                    'Iterations': itr,
                                    },
                                    {
                                    '* Accuracy/train': acc,
                                    '* Accuracy/val': val_acc,
                                    '* Loss/train': total_loss/p_itr,
                                    '* Loss/val': val_total_loss/val_check_period,
                                    })
                                # writer.add_scalar('Loss/compare', 
                                #                   {'train': total_loss/p_itr,
                                #                    'val': val_total_loss/val_check_period}, itr)
                                # writer.add_scalar('Accuracy/compare', 
                                #                   {'train': acc,
                                #                    'val': val_acc}, itr)
            
                    total_loss = 0
                    val_total_loss = 0
                    data_available = True

                itr += 1
                 
            if show_pyplot:
                plt.plot(loss_list, label='loss')
                if val_sample or val_check_period:
                    plt.plot(val_loss_list, label='val_loss')
                plt.legend()
                plt.title('training and val loss')
                plt.show()
                ###
                plt.plot(acc_list, label='accuracy')
                if val_sample or val_check_period:
                    plt.plot(val_acc_list, label='val_accuracy')
                plt.legend()
                plt.title('training and val accuracy')
                plt.show()
                
            ''' Saving multiple weights'''  
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            checkpoint_dir = f'{weight_dir_path}/checkpoint'
            best_model_dir = f'{weight_dir_path}/best_model_dir' 
            filename_pth = f'ckpt_densenet121_mark_correct_{self.img_size}_s3_{itr}.pth'
            
            if itr == int(epochs*len(dataloader_train)/10):
            # if epoch % 10 == 0:
                save_ckp(state=checkpoint, 
                        checkpoint_dir=checkpoint_dir, 
                        best_model_dir=best_model_dir,
                        filename_pth=filename_pth)
                # torch.save(self.model.state_dict(), os.path.join(weight_dir_path, filename_pth))
                printj.yellow.on_purple(f"Weight file saved: {os.path.join(weight_dir_path, filename_pth)}")
        print("Total iterations: ", itr-1)

        if show_pyplot:
            plt.plot(loss_list, label='loss')
            plt.plot(acc_list, label='accuracy')
            plt.legend()
            plt.title('training loss and accuracy')
            plt.show()

            if val_sample or val_check_period:
                plt.plot(val_acc_list, label='Test_accuracy')
            plt.plot(acc_list, label='Train_accuracy')
            plt.legend()
            plt.title('training loss and accuracy')
            plt.show()

        # filename_pth = 'ckpt_densenet121_catdog.pth'
        # filename_pth = f'ckpt_densenet121_mark_exist_{self.img_size}.pth'
        filename_pth = f'ckpt_densenet121_mark_correct_{self.img_size}_s3.pth'
        torch.save(self.model.state_dict(), os.path.join(weight_dir_path, filename_pth))

    def _get_something(self, data=None):
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        # testset = CatDogDataset(test_files, TEST_IMG_DIR_PATH, mode='test', transform = test_transform)
        # testloader = DataLoader(testset, batch_size = 32, shuffle=False, num_workers=4)
        if data is None:
            dataset = self.get_dataset()
            data = dataset["train"]
        testloader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        self.model.eval()
        fn_list = []
        pred_list = []
        for x, fn in testloader:
            with torch.no_grad():
                x = x.to(self.device)
                output = self.model(x)
                pred = torch.argmax(output, dim=1)
                fn_list += [n[:-4] for n in fn]
                pred_list += [p.item() for p in pred]

        submission = pd.DataFrame({"id": fn_list, "label": pred_list})
        submission.to_csv(f'preds_densenet121_dir_{self.img_size}.csv',
                          #   index=False
                          )

    def get_pred(self, model, model_path, test_dir_path, no_of_samples: int = 24, test_label: int = 1, save_csv_path: str = None):
        model.load_state_dict(torch.load(model_path))
        # "/home/jitesh/prj/classification/test/bolt/ckpt_densenet121_mark_correct_128_s3_5.pth"))
        data_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
        test_list = [file for file in os.listdir(test_dir_path) if os.path.isfile(
            os.path.join(test_dir_path, file)) and "b1" in file]
        test_list = sorted(test_list)
        # print(test_list)
        test_data = BoltDataset(test_list, test_dir_path, mode="test",
                                transform=data_transform, test_label=test_label)
        testloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        model.eval()
        fn_list = []
        pred_list = []
        for x, fn in testloader:
            with torch.no_grad():
                x = x.to(self.device)
                output = model(x)
                pred = torch.argmax(output, dim=1)
                fn_list += [n[:-4] for n in fn]
                pred_list += [p.item() for p in pred]

        submission = pd.DataFrame({"id": fn_list, "label": pred_list})
        if save_csv_path is None:
            save_csv_path = f'preds_densenet121_dir_{self.img_size}_test_.csv'
        submission.to_csv(save_csv_path,
                          #   index=False
                          )
        samples, _ = iter(testloader).next()
        samples = samples.to(self.device)
        val_sample = samples[:no_of_samples]
        return val_sample
    
    
    def _get_pred_(self, data=None):
        if data is None:
            dataset = self.get_dataset()
            data = dataset["train"]
        testloader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        samples, _ = iter(testloader).next()
        samples = samples.to(self.device)
        fig = plt.figure(figsize=(24, 16))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor('xkcd:salmon')
        ax.set_facecolor((1.0, 0.47, 0.42))
        fig.tight_layout()
        output = self.model(samples[:24])
        pred = torch.argmax(output, dim=1)
        pred = [p.item() for p in pred]
        # ad = {0:'cat', 1:'dog'}
        # ad = {0:'no mark', 1:'mark'}
        ad = {0: 'Incorrect', 1: 'Correct'}
        # for num, sample in enumerate(samples[:24]):
        for num, sample in enumerate(samples[:24]):
            plt.subplot(4, 6, num+1)
            plt.title(ad[pred[num]])
            # plt.axis('off')
            sample = sample.cpu().numpy()
            plt.imshow(np.transpose(sample, (1, 2, 0)))
        # ax = plt.gca()
        plt.savefig('inference_mark_direction.png')


    def preview_aug(self, image_path, grid=[4, 4], save_path=None):
        pillow_image = Image.open(image_path)
        image = np.array(pillow_image)
        images = []
        new_im = Image.new('RGB', (grid[1]*self.img_size,grid[0]*self.img_size))
        # bolt =BoltClassifier(img_size=img_size)
        transform = self.get_augmentation() #load_path='/home/jitesh/prj/classification/test/bolt_src/aug/aug_seq.json')
        
        # rsize = self.img_size
        # rsize = int(1000/max(grid[0], grid[1]))
        # xp = int(1000/grid[1])
        # yp = int(1000/grid[0])
        for i in range(grid[0]):
            for j in range(grid[1]):
                transformed_image = transform(image=image)['image']
                # transformed_image.thumbnail((xp,xp))
                t_image = self.convert_img_shape_aug_to_normal(transformed_image)
                # transformed_image.resize(xp,xp, Image.ANTIALIAS)
                # print(transformed_image)
                # new_im.paste(transformed_image, (i*xp,j*xp))
                # t_image = t_image.resize((rsize,rsize), Image.ANTIALIAS)
                # print(t_image)
                new_im.paste(t_image, (i*self.img_size,j*self.img_size))
                # new_im.paste(t_image)
                # import sys
                # sys.exit()
        new_im.show()
        if save_path:
            new_im.save(save_path)

