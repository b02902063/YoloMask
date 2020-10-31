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
import pickle
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
    bbox = []
    for ann in anns:
        if ann["area"] == 0:
            continue
        cls = categories[ann["category_id"]]
        b = ann["bbox"]
        b.append(cls)
        if b in bbox:
            continue
           
        bbox.append(b)
        mask = annToMask(ann, img_size)
        #mask *= (cls + 1)
        output.append(mask)
    
    if len(output) > 0:
        output = np.stack(output, axis=-1).astype(np.uint8)
    else:
        output = np.empty(0)
    # save as compressed npz
    np.savez_compressed(output_semantic, mask=output)
    # Image.fromarray(output).save(output_semantic)


def create_coco_semantic_from_instance(instance_json, sem_seg_root, categories):
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
    os.makedirs(sem_seg_root, exist_ok=True)

    coco_detection = COCO(instance_json)

    def iter_annotations():
        for img_id in coco_detection.getImgIds():
            anns_ids = coco_detection.getAnnIds(img_id, iscrowd=False)
            anns = coco_detection.loadAnns(anns_ids)
            img = coco_detection.loadImgs(int(img_id))[0]
            file_name = os.path.splitext(img["file_name"])[0]

            output = os.path.join(sem_seg_root, file_name + '.npz')
            yield anns, output, img

    images_root = sem_seg_root.replace("segmentations", "images")
    def iter_annotations_for_polygon():
        for img_id in coco_detection.getImgIds():
            anns_ids = coco_detection.getAnnIds(img_id, iscrowd=False)
            anns = coco_detection.loadAnns(anns_ids)
            img = coco_detection.loadImgs(int(img_id))[0]
            file_name = os.path.splitext(img["file_name"])[0]

            output = os.path.join(images_root, file_name + '.jpg')
            output = os.path.join("data", output).replace('/', os.sep)
            yield anns, output, img
    # single process
    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    for anno, oup, img in iter_annotations():
        _process_instance_to_semantic(
            anno, oup, img, categories)
    print("Finished. time: {:.2f}s".format(time.time() - start))
    return
    whole_dict = {}
    for anno, oup, img in iter_annotations_for_polygon():
        h, w = (img["height"], img["width"])
        all_ann = []
        for ann in anno:
            segm = ann['segmentation']
            new_ann = []
            for a in segm:
                for i in range(len(a)):
                    if i % 2 == 1:
                        a[i] /= h
                    else:
                        a[i] /= w
                new_ann.append(np.array(a))
            all_ann.append(np.array(new_ann))
        whole_dict[oup] = all_ann
    output_path = os.path.join(sem_seg_root, "output.pickle")
    with open(output_path, 'wb') as handle:
        pickle.dump(whole_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return
    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(
            _process_instance_to_semantic,
            categories=categories),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))
    



def get_parser():
    parser = argparse.ArgumentParser(description="Keep only model in ckpt")
    parser.add_argument(
        "--dataset-name",
        default="coco",
        help="dataset to generate",
    )
    return parser


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "coco")
    thing_id_to_contiguous_id = _get_coco_instances_meta()["thing_dataset_id_to_contiguous_id"]
    split_name = 'train2017'
    annotation_name = "annotations/instances_{}.json"

    for s in ["val2017", "train2017"]:
        create_coco_semantic_from_instance(
            os.path.join(dataset_dir, "annotations/instances_{}.json".format(s)),
            os.path.join(dataset_dir, "segmentations/{}".format(s)),
            thing_id_to_contiguous_id
        )
