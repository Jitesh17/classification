import os
from sys import exit as x
from time import time

import printj
from torch.utils.tensorboard import SummaryWriter

from TheLib import BoltClassifier

def train(img_size, batch_size, learning_rate, 
          tensorboard_log_dir_path,epochs,val_check_period,
          iteration_limit,weight_dir_path,
          ):
    printj.black.bold_on_white(f"Training with\n\
                                batch_size: {batch_size}\n\
                                learning_rate: {learning_rate}")
    bolt = BoltClassifier(device= 'cuda', img_size=img_size, batch_size=batch_size, data_types= ["train", "val", "test"],
                    dataset_path= "/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped",
                    b_type_list=['b10', 'b11'], num_workers=16)
    img_path_list = bolt.split_train_test(verbose=False)
    # bolt.view_data2()
    bolt.set_model(learning_rate = learning_rate, 
                #    scheduler_on=True, 
                scheduler_steps=[2000])
    bolt.train(epochs=epochs, 
            val_check_period=val_check_period, 
            show_tensorboard=show_tensorboard, 
            tensorboard_log_dir_path=tensorboard_log_dir_path, 
            show_pyplot=False, 
            weight_dir_path=weight_dir_path,
            iteration_limit=iteration_limit,
            )
# img_size=128
# bolt = BoltClassifier(device= 'cuda', img_size=img_size, batch_size=16, data_types= ["train", "val", "test"],
#                 dataset_path= "/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped",
#                 b_type_list=['b10', 'b11'], num_workers=16)


"""
# bolt.get_pred()
bolt.preview_aug(
    image_path="/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped/b11/000008.png",
    grid=[16, 16],
    save_path="/home/jitesh/prj/classification/test/bolt_src/aug_images/aug2.png"
    )
"""

# print(bolt.img_size)

# val_sample = bolt.get_val(bolt.model, model_path=)

# bolt.show_img("/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped/b10/000003.png")
# img_path_list = bolt.split_train_test()
# filename_list = bolt.get_image_list()
# aug_seq = bolt.get_augmentation()
# bolt.view_data()
# dataset = bolt.get_dataset()
# dataloader = bolt.get_dataloader(dataset)
# bolt.view_data2()
img_size=128
batch_size = 32
learning_rate = 0.1
show_tensorboard = True
# tensorboard_log_dir_path = f'runs3/rw_s{img_size}_lr{learning_rate}_no_scheduler_val_10_{int(time())}'

  
train(img_size=img_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tensorboard_log_dir_path=f'runs3/rw_s{img_size}_bs{batch_size}_lr{learning_rate}_no_scheduler_val_{int(time())}',
        epochs=50, 
        iteration_limit=10000,
        val_check_period=batch_size,
        weight_dir_path="/home/jitesh/prj/classification/test/bolt_src/weights_b32_l0.1"
        )    
x()     
for batch_size in [64, 32, 16 ]:
    for learning_rate in [0.1, 0.01, 0.001 ]:
        train(img_size=img_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              tensorboard_log_dir_path=f'runs4/rw_s{img_size}_bs{batch_size}_lr{learning_rate}_no_scheduler_val_{int(time())}',
              epochs=50, 
              iteration_limit=10000,
              val_check_period=batch_size,
        
              )

"""
"""