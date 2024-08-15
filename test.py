import json 
import numpy as np 
f = open('/home/qdinh/FasterRCNN-PyTorch/bbox_results/test2017seen_bbox_results.json')
f = json.load(f)

gt = open('/home/qdinh/data/DOTA/annotations/instances_test2017seen.json')
gt = json.load(gt)

true = [x for x in gt['annotations'] if x['image_id'] == 5]
pred = [x for x in f if x['image_id'] == 5 and x['score'] > 0.5]

true_bbox = np.array([x['bbox'] + [x['category_id']] for x in gt['annotations'] if x['image_id'] == 5])
pred_bbox = np.array([x['bbox'] + [x['category_id']] for x in f if x['image_id'] == 5 and x['score'] > 0.5])
gt['annotations'][2]