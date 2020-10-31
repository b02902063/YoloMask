# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
import functools
import multiprocessing as mp
import numpy as np
import os
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import sys
import cv2

from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta


def annToRLE(ann, img_size):
    h, w = img_size
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annToMask(ann, img_size):
    rle = annToRLE(ann, img_size)
    m = maskUtils.decode(rle)
    return m


def _process_instance_to_semantic(anns, output_semantic, img, categories):
    img_size = (img["height"], img["width"])
    output = []
    for ann in anns:
        mask = annToMask(ann, img_size)
        mask *= (categories[ann["category_id"]] + 1)
        output.append(mask)
    
    if len(output) > 0:
        output = np.stack(output, axis=0)
    else:
        output = np.empty(0)
    # save as compressed npz
    np.savez_compressed(output_semantic, mask=output)
    # Image.fromarray(output).save(output_semantic)


def draw(instance_json, sem_seg_root, categories, img_id):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.

    It maps all thing categories to contiguous ids starting from 1, and maps all unlabeled pixels to class 0

    Args:
        instance_json (str): path to the instance json file, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (dict): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    coco_detection = COCO(instance_json)

    anns_ids = coco_detection.getAnnIds(img_id, iscrowd=False)
    anns = coco_detection.loadAnns(anns_ids)
    print(anns[2])
    print(anns[2]["area"] == 0)
    img_file = str(img_id).zfill(12) + ".jpg"
    img = cv2.imread("E:/YOLO-HarDNet_/data/coco/images/train2017/" + img_file)
    iw = img.shape[1]
    ih = img.shape[0]
    bbox = []
    for ann in anns:
        x,y,w,h  = ann["bbox"]
        #print(np.round((x + w/2)/iw, decimals=6), \
        #np.round((y + h/2)/ih, decimals=6), np.round(w/iw, decimals=6) ,np.round(h/ih, decimals=6))
        x2 = int(x+w)
        y2 = int(y+h)
        x = int(x)
        y = int(y)
        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        
    cv2.imwrite("test.jpg", img)



def get_parser():
    parser = argparse.ArgumentParser(description="Keep only model in ckpt")
    parser.add_argument(
        "--dataset-name",
        default="coco",
        help="dataset to generate",
    )
    return parser


if __name__ == "__main__":
    img_id = int(sys.argv[1])
    dataset_dir = os.path.join(os.path.dirname(__file__), "coco")
    thing_id_to_contiguous_id = _get_coco_instances_meta()["thing_dataset_id_to_contiguous_id"]
    split_name = 'train2017'
    annotation_name = "annotations/instances_{}.json"

    for s in ["train2017"]:
        draw(
            os.path.join(dataset_dir, "annotations/instances_{}.json".format(s)),
            os.path.join(dataset_dir, "segmentations/{}".format(s)),
            thing_id_to_contiguous_id,
            img_id
        )
