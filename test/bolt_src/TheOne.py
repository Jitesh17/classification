from TheLib import BoltClassifier


bolt = BoltClassifier(device= 'cuda', img_size=256, batch_size=8, data_types= ["train", "val", "test"],
                dataset_path= "/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped",
                b_type_list=['b10', 'b11'], num_workers=8)

"""
# bolt.get_pred()
bolt.preview_aug(
    image_path="/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped/b11/000008.png",
    grid=[16, 16],
    save_path="/home/jitesh/prj/classification/test/bolt_src/aug_images/new_aug.png"
    )
"""

# print(bolt.img_size)

# val_sample = bolt.get_val(bolt.model, model_path=)

# bolt.show_img("/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped/b10/000003.png")
# img_path_list = bolt.split_train_test()
filename_list = bolt.get_image_list()
# aug_seq = bolt.get_augmentation()
# bolt.view_data()
# dataset = bolt.get_dataset()
# dataloader = bolt.get_dataloader(dataset)
# bolt.view_data2()
bolt.set_model(learning_rate = 0.001, 
            #    scheduler_on=True, 
               scheduler_steps=[2000])
bolt.train(epochs=500, 
           show_tensorboard=True, 
           tensorboard_log_dir_path='runs/s8_lr0.001_no_scheduler_1', 
           show_pyplot=False, 
           weight_dir_path="/home/jitesh/prj/classification/test/bolt_src/weights")

