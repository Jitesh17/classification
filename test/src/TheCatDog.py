from TheLib import BoltClassifier
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random
import os
from time import time 

# filenames = os.listdir("/home/jitesh/3d/data/unstructured_data/dogs-vs-cats/train")
# categories = []
# for filename in filenames:
#     category = filename.split('.')[0]
#     if category == 'dog':
#         categories.append(1)
#     else:
#         categories.append(0)

# df = pd.DataFrame({
#     'filename': filenames,
#     'category': categories
# })
img_size=128
bolt = BoltClassifier(device= 'cuda', img_size=img_size, batch_size=16, data_types= ["train", "val"],
                dataset_path= "/home/jitesh/3d/data/unstructured_data/dogs-vs-cats",
                b_type_list=['cat', 'dog'], num_workers=16)

"""
# bolt.get_pred()
bolt.preview_aug(
    image_path="/home/jitesh/3d/data/unstructured_data/dogs-vs-cats/test/82.jpg",
    grid=[16, 16],
    save_path="/home/jitesh/prj/classification/test/bolt_src/aug_images/cat_aug.png"
    )
"""

# print(bolt.img_size)

# val_sample = bolt.get_val(bolt.model, model_path=)

# bolt.show_img("/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped/b10/000003.png")
img_path_list = bolt.split_train_test(split_ratio=[249/250, 1/250, 0])
# filename_list = bolt.get_image_list()
# aug_seq = bolt.get_augmentation()
# bolt.view_data()
# dataset = bolt.get_dataset()
# dataloader = bolt.get_dataloader(dataset)
# bolt.view_data2()
learning_rate = 0.01
bolt.set_model(learning_rate = learning_rate, 
            #    scheduler_on=True, 
               scheduler_steps=[2000])
bolt.train(epochs=500,
           val_check_period=16, 
           show_tensorboard=True, 
           tensorboard_log_dir_path=f'runs2/cdp_rw_s{img_size}_lr{learning_rate}_no_scheduler_val_10_{int(time())}', 
           show_pyplot=False, 
           weight_dir_path="/home/jitesh/prj/classification/test/bolt_src/catdog_weights")
""""""
# v= []
# for i in range (9):
#     v=v+[i, i+1]
#     print( v)
    
# print(v)