import os, sys
import json
from pyjeasy.file_utils import make_dir_if_not_exists
from pyjeasy.image_utils.output import show_image
from pyjeasy.image_utils.draw import draw_bbox
from jaitool import annotation as a 
from annotation_utils.ndds.structs import NDDS_Dataset
from annotation_utils.coco.structs import COCO_Dataset, COCO_Category_Handler, COCO_Category
import printj
from common_utils.common_types import BBox
import cv2
from tqdm import tqdm


def run(path):
    printj.yellow(f'Processing {path}')
    ndds_dataset = NDDS_Dataset.load_from_dir(
        json_dir=path,
        show_pbar=True
    )

    for frame in tqdm(ndds_dataset.frames):
        # printj.red(frame)
        # Fix Naming Convention
        for ann_obj in frame.ndds_ann.objects:
            if ann_obj.class_name.startswith('bolt'):
                bolt_bbox = ann_obj.bounding_box
            roi_bbox = bolt_bbox
        for ann_obj in frame.ndds_ann.objects:
            if ann_obj.class_name.startswith('i-'):
                imark_bbox = ann_obj.bounding_box
                roi_bbox = roi_bbox + imark_bbox
            elif ann_obj.class_name.startswith('o-'):
                omark_bbox = ann_obj.bounding_box
                # printj.purple(roi_bbox)
                roi_bbox = roi_bbox + omark_bbox
                # printj.cyan(roi_bbox)
                # printj.cyan(omark_bbox)
                # printj.green('ghcfxnn')
        roi_bbox = roi_bbox.to_int()
        # printj.red(roi_bbox)
        # printj.red(f"{roi_bbox.xmin}:{roi_bbox.xmax}, {roi_bbox.ymin}:{roi_bbox.ymax}")
        input_path = frame.img_path
        output_path = os.path.abspath(f"{input_path}/../../../bolt_cropped")\
            + "/"\
            + os.path.abspath(f"{input_path}").split('/')[-2]\
            + "/"\
            + os.path.abspath(f"{input_path}").split('/')[-1]
        make_dir_if_not_exists(os.path.abspath(f"{output_path}/../../.."))
        make_dir_if_not_exists(os.path.abspath(f"{output_path}/../.."))
        make_dir_if_not_exists(os.path.abspath(f"{output_path}/.."))
        img = cv2.imread(input_path)
        
        cv2.imwrite(output_path, img[roi_bbox.ymin:roi_bbox.ymax, roi_bbox.xmin:roi_bbox.xmax])
        # printj.green(f'Wrote {output_path}')
        
        # img = draw_bbox(img, omark_bbox.to_int().to_list())
        # img = draw_bbox(img, roi_bbox.to_int().to_list())
        # show_image(img)
        # if show_image(img[roi_bbox.ymin:roi_bbox.ymax, roi_bbox.xmin:roi_bbox.xmaxn]):
        #     break
if __name__ == "__main__":
    outer_dir = "/home/jitesh/3d/data/UE_training_results/bolt3/detailed_bolt"
    # SRC_dir = "/home/jitesh/3d/data/UE_training_results/bolt2/bolt_detailed/b00 (copy)"
    # SRC_dir = "/home/jitesh/3d/data/UE_training_results/bolt2/bolt_detailed/b10"
    
    # for i, path in enumerate(os.listdir(outer_dir)):
    for path in tqdm(os.listdir(outer_dir)):
        # if i < 2:
        #     continue
        run(os.path.join(outer_dir, path))