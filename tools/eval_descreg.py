import torch
import numpy as np
import random
import os
import yaml
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from model.faster_rcnn import FasterRCNN
from dataset.voc import VOCDataset
import matplotlib.pyplot as plt 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_dataset(args):
    # Read the config file #

    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    dataset_test = VOCDataset('test', im_dir=dataset_config['im_test_seen_path'], ann_dir=dataset_config['ann_test_seen_path'], is_zsd = True)
    # test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['ckpt_path'],args['ckpt']),map_location=device))
    return faster_rcnn_model, dataset_test

def predict_coco(detector, dataset_test, pred_style, threshold = 0.05):
    results = []
    detector.eval()
    with torch.no_grad():
        for data in tqdm(dataset_test):
            im, target, fname, im_id = data

            im = im.float().to(device).unsqueeze(0)
            rpn_output, frcnn_output = detector(im, target = None, pred_style = pred_style)
            
            boxes = frcnn_output['boxes'].detach().cpu().numpy()
            labels = frcnn_output['labels'].detach().cpu().numpy()
            scores = frcnn_output['scores'].detach().cpu().numpy()

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                for idx, box in enumerate(boxes):
                        score = float(scores[idx])
                        label = int(labels[idx])
                        label = dataset_test.idxzsd2idx[label] #convert from zsd index to index 
                        box = boxes[idx, :]
                        # break
                        # scores are sorted, so we can break
                        if score < threshold:
                            break

                        # append detection for each positively labeled class
                        image_result = {
                            'image_id'    : im_id,
                            'category_id' : int(label),
                            'score'       : float(score),
                            'bbox'        : box.tolist(),
                        }

                        # append detection to results
                        results.append(image_result)

        if not len(results):
            print('No predictions!')
            return

        # write output
        os.makedirs('./bbox_results/', exist_ok= True)
        json.dump(results, open('./bbox_results/{}_bbox_results.json'.format(dataset_test.set_name), 'w'), indent=4)

def evaluate_coco(  dataset_test, data_path = '/home/qdinh/data', draw_PRcurves = False, return_classap = True):
    # load results in COCO evaluation tool
    # coco_true = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
    coco_true = COCO(os.path.join(f'{data_path}/{dataset_test.dataset_name}', 'annotations', 'instances_' + dataset_test.set_name + '.json'))
    coco_pred = coco_true.loadRes('./bbox_results/{}_bbox_results.json'.format(dataset_test.set_name))
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # breakpoint()
    # Extract class-wise average precision
    class_ap = {}
    for i, catId in enumerate(coco_true.getCatIds()):
        try:
            class_ap[coco_true.loadCats(catId)[0]['name']] = coco_eval.eval['precision'][0, :, i, 0, 2].mean()
        except:
            class_ap[coco_true.loadCats(catId)[0]['name']] = 0.

    if draw_PRcurves == True:
        # Draw AP@0.5 PR curves
        precision = coco_eval.eval['precision']
        recall = coco_eval.params.recThrs

        # Get category IDs and names
        cat_ids = coco_true.getCatIds()
        cat_names = [cat['name'] for cat in coco_true.loadCats(cat_ids)]
        seen_ids = [dataset_test.classes.index(label) for label in dataset_test.seen_classes]
        # Plot precision-recall curve for each category
        plt.figure(figsize=(10, 6))
        for i, cat_id in enumerate(cat_ids):
            if cat_id in seen_ids:
                # Extract precision for the current category, averaged over IoU thresholds, area ranges, and max detections
                precision_cat = precision[0, :, i, 0, 2]  # shape (recThrs,)
                plt.plot(recall, precision_cat, label='{}. AP: {:.1f}'.format(cat_names[i], np.mean(precision_cat)*100))
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve of Seen classes')
                plt.grid(True)
                plt.legend()
        os.makedirs('./figures', exist_ok=True)
        plt.savefig(f'./figures/Seen PR curves.png')

    # plt.figure(figsize=(10, 6))
    # for i, cat_id in enumerate(cat_ids):
    #     if cat_id in dataset.seen_ids:
    #         # Extract precision for the current category, averaged over IoU thresholds, area ranges, and max detections
    #         precision_cat = precision[0, :, i, 0, 2]  # shape (recThrs,)
    #         plt.plot(recall, precision_cat, label='{}. AP: {:.1f}'.format(cat_names[i], np.mean(precision_cat)*100))
    #         plt.xlabel('Recall')
    #         plt.ylabel('Precision')
    #         plt.title('Precision-Recall Curve of seen classes')
    #         plt.grid(True)
    #         plt.legend()
    # plt.savefig(f'./figures/Seen PR curves.png')

    if return_classap == True:
        return coco_eval.stats, class_ap
    else:
        return coco_eval.stats

