import os
from os import path
from os.path import join
import sys
import json
import printj
import pyjeasy.file_utils as f
import cv2
from tqdm import tqdm

PATH = "/home/jitesh/sekisui/bolt/hexagon_bolts"
OUTPUT_PATH = "/home/jitesh/sekisui/bolt/cropped_hexagon_bolts"
f.make_dir_if_not_exists(OUTPUT_PATH)

json_list = f.get_all_filenames_of_extension(dirpath=PATH, extension="json")
# printj.blue(json_list)

for json_file in tqdm(json_list):
    filename = json_file.split(".")[0]
    output_image_path = os.path.join(OUTPUT_PATH, f"{filename}.jpg")
    image_path = os.path.join(PATH, f"{filename}.jpg")
    json_path = os.path.join(PATH, json_file)
    
    if f.path_exists(image_path):
        img = cv2.imread(image_path)
        with open(json_path) as json_data:
            data = json.load(json_data)
        data = data["shapes"]
        i = 0
        for d in data:
            if d["label"] =="bolt-roi":
                [p1, p2] = d["points"]
                xmin = int(min(p1[0], p2[0]))
                ymin = int(min(p1[1], p2[1]))
                xmax = int(max(p1[0], p2[0]))
                ymax = int(max(p1[1], p2[1]))
            output_img = img[ymin:ymax, xmin:xmax]
            # cv2.imshow("", output_img)
            # cv2.waitKey(0)
            
            output_image_path = os.path.join(OUTPUT_PATH, f"{filename}_{i}.jpg")
            cv2.imwrite(output_image_path, output_img)
            i += 1
            # printj.cyan(filename)
            # sys.exit()