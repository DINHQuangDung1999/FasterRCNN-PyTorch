import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET


def load_images_and_anns(im_dir, ann_dir, label2idx):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_dir: Path of the images
    :param ann_dir: Path of annotation xmlfiles
    :param label2idx: Class Name to index mapping for dataset
    :return:
    """
    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.xml'))):
        im_info = {}
        im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
        im_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(im_info['img_id']))
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        im_info['width'] = width
        im_info['height'] = height
        detections = []
        
        for obj in ann_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text))-1,
                int(float(bbox_info.find('ymin').text))-1,
                int(float(bbox_info.find('xmax').text))-1,
                int(float(bbox_info.find('ymax').text))-1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        im_info['detections'] = detections
        im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


class VOCDataset(Dataset):
    def __init__(self, split, dataset_name, im_dir, ann_dir, is_zsd = False):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        self.dataset_name = dataset_name
        self.set_name = im_dir.split('/')[-1]

        if is_zsd == True:
            if self.dataset_name == 'PascalVOC':
                self.seen_classes   = sorted(['person', 'bird', 'cat', 'cow', 'horse', 'sheep',
                                        'aeroplane', 'bicycle', 'boat', 'bus','motorbike', 
                                        'bottle', 'chair', 'diningtable', 'pottedplant','tvmonitor' ])
                self.unseen_classes = sorted(['car', 'dog', 'sofa', 'train'])

                classes_zsd = ['background'] + self.seen_classes + self.unseen_classes
                self.classes_zsd    = classes_zsd
            if self.dataset_name == 'DOTA':
                self.seen_classes   = sorted(['plane', 'ship', 'storage-tank', 'baseball-diamond', 'basketball-court', 
                                              'ground-track-field', 'harbor', 'bridge', 'large-vehicle', 
                                              'small-vehicle', 'roundabout', 'container-crane'])
                self.unseen_classes = sorted(['helicopter', 'soccer-ball-field', 'swimming-pool', 'tennis-court'])

                classes_zsd = ['background'] + self.seen_classes + self.unseen_classes
                self.classes_zsd    = classes_zsd
            if self.dataset_name == 'DIOR':
                self.seen_classes   = sorted(['airplane', 'baseballfield', 'bridge', 'chimney', 'dam', 
                                              'Expressway-Service-area', 'Expressway-toll-station', 
                                              'golffield', 'harbor', 'overpass', 'ship', 'stadium', 
                                              'storagetank', 'tenniscourt', 'trainstation', 'vehicle'])
                self.unseen_classes = sorted(['airport', 'basketballcourt', 'groundtrackfield', 'windmill'])

                classes_zsd = ['background'] + self.seen_classes + self.unseen_classes
                self.classes_zsd    = classes_zsd

            classes = sorted(self.classes_zsd[1:])
            classes = ['background'] + classes
            self.classes    = classes

        self.label2idx  = {classes[idx]: idx for idx in range(len(classes))} # if is_zsd == True then this maps to zsd indices
        self.idx2label  = {idx: classes[idx] for idx in range(len(classes))}
        
        self.idx_to_idxzsd = {self.classes.index(label): self.classes_zsd.index(label) \
                              for label in self.classes}
        self.idxzsd_to_idx = {idx_zsd: idx for idx, idx_zsd in self.idx_to_idxzsd.items()}
        # breakpoint()
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, self.label2idx)
    
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([self.idx_to_idxzsd[detection['label']] for detection in im_info['detections']])
        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return im_tensor, targets, im_info['filename']
        