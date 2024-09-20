from __future__ import print_function, division

import os
import torch
import numpy as np
import random

import torchvision
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

from PIL import Image


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, split, dataset_name, root_dir, set_name, is_zsd = True):
        ''' Added parameters seen for zero-shot training '''
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.split = split
        self.root_dir = root_dir
        self.set_name = set_name
        self.dataset_name = dataset_name
        # breakpoint()
        self.coco      = COCO(os.path.join(self.root_dir, dataset_name, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.is_zsd = is_zsd
        self.load_classes()
        
    def load_classes(self):
        
        if self.is_zsd == True:
            print(f'Mode: zsd')
            if self.dataset_name == 'DOTA': # 0 represents background
                self.seen_ids = [0,1,2,3,4,6,7,8,9,10,11,12,16]
                self.unseen_ids = [5,13,14,15] # helicopter, soccerball-field, swimming-pool, tennis-court
            elif self.dataset_name == 'PascalVOC':
                self.seen_ids = [0,1,2,3,4,5,6,8,9,10,11,13,14,15,16,17,20]
                self.unseen_ids = [7,12,18,19] # car, dog, sofa, train
            elif self.dataset_name == 'xView':
                self.seen_ids = [0, 1, 2, 3, 5, 6, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 41, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60]
                self.unseen_ids = [4, 7, 8, 12, 24, 25, 28, 37, 39, 42, 44, 57]                
            elif self.dataset_name == 'DIOR':
                self.seen_ids = [0,1,3,5,6,7,8,9,10,12,13,14,15,16,17,18,19]
                self.unseen_ids = [2,4,11,20] # airport, basketballcourt, groundtrackfield, windmill
            ''' Load seen&unseen class'''

            self.idx2idxzsd          = {}
            self.idxzsd2idx          = {}
            for i, id in enumerate(self.seen_ids + self.unseen_ids):
                self.idxzsd2idx[i] = id
                self.idx2idxzsd[id] = i

            categories = self.coco.loadCats(self.coco.getCatIds())
            categories.sort(key=lambda x: x['id'])
            categories = [{'supercategory': 'background', 'id': 0, 'name': 'background'}] + categories
            self.label2idx             = {}
            self.idx2label             = {}
            for c in categories: 
                self.label2idx[c['name']] = c['id'] 
                self.idx2label[c['id']] = c['name']

        else:
            print(f'Mode: traditional')
            ''' Load seen all classes'''
            # load class names (name -> label)
            categories = self.coco.loadCats(self.coco.getCatIds())
            categories.sort(key=lambda x: x['id'])
            categories = [{'supercategory': 'background', 'id': 0, 'name': 'background'}] + categories

            self.label2idx           = {}
            self.idx2label           = {}
            for c in categories:
                self.label2idx[c['name']] = c['id']
                self.idx2label[c['id']] = c['name'] 
        # breakpoint()
            
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        path       = os.path.join(self.root_dir, self.dataset_name, 'images', self.set_name, image_info['file_name'])
        im = Image.open(path)
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = self.load_annotations(idx)
        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])

        return im_tensor, targets, path, image_info['id']

    def load_annotations(self, image_index):
        
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        targets = {}
        targets['bboxes'] = []
        targets['labels'] = []
        
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return targets

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            bbox = a['bbox']
            ''' Modify this part to obtain label for seen id''' #id [1,2,...,16] to label [0,1,...,13]
            # annotation[0, 4]  = self.coco_label_to_label(a['category_id']) # already mapped from (1,...,16) to (0,...,15)
            # annotation[0, 4]  = self.seen_coco_labels[a['category_id']]
            label  = self.idx2idxzsd[a['category_id']]
            targets['bboxes'].append(bbox)
            targets['labels'].append(label)

        targets['bboxes'] = np.array(targets['bboxes'])
        targets['labels'] = np.array(targets['labels'])
        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        targets['bboxes'][:, 2] = targets['bboxes'][:, 0] + targets['bboxes'][:, 2]
        targets['bboxes'][:, 3] = targets['bboxes'][:, 1] + targets['bboxes'][:, 3]

        targets['bboxes'] = torch.as_tensor(targets['bboxes']).float()
        targets['labels'] = torch.as_tensor(targets['labels']).long()
        # breakpoint()
        return targets

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        '''Modified to get the correct number of classes'''
        # return 80
        if self.is_zsd == True:
            return len(self.seen_ids), len(self.unseen_ids)
        else:
            return len(self.classes)
