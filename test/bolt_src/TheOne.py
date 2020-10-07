from TheLib import BoltClassifier 

bolt = BoltClassifier(device= 'cuda', img_size=128, batch_size=8, data_types= ["train", "val", "test"],
                dataset_path= "/home/jitesh/3d/data/UE_training_results/bolt3/bolt_cropped",
                b_type_list=['b10', 'b11'], num_workers=2)


# print(bolt.img_size)

val_sample = bolt.get_val(bolt.model, )