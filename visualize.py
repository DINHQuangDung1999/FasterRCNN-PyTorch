import torch
import numpy as np
import cv2
import argparse
import random
import os
os.chdir('./FasterRCNN-PyTorch')
import yaml
import shutil
from easydict import EasyDict
from tqdm import tqdm
from model.faster_rcnn import FasterRCNN
from dataset.voc import VOCDataset


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
    
    detector = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    detector.eval()
    detector.to(device)
    detector.load_state_dict(torch.load(os.path.join(train_config['ckpt_path'],args['ckpt']),map_location=device))
    return detector, dataset_test

args = EasyDict({'config_path':'config/voc_seen.yaml',
                    'ckpt': 'frcnn_seen_19.pt'})
detector, dataset_test = load_model_and_dataset(args)

def infer():
    if os.path.exists('./dump'):
        shutil.rmtree('./dump')
    os.makedirs('./dump', exist_ok=True)

    # Hard coding the low score threshold for inference on images for now
    # Should come from config
    detector.roi_head.low_score_threshold = 0.7
    
    for sample_count in tqdm(range(100)):
        random_idx = random.randint(0, len(dataset_test))
        im, target, fname = dataset_test[random_idx]
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)

        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            text = dataset_test.idx2label[target['labels'][idx].detach().cpu().item()]
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=1, color=[0, 255, 0])
            # cv2.putText(gt_im, text=text,
            #             org=(x1+5, y1+15),
            #             thickness=1,
            #             fontScale=1,
            #             color=(255, 255, 255),
            #             fontFace=cv2.FONT_HERSHEY_PLAIN)
        
        # Getting predictions from trained model
        rpn_output, frcnn_output = detector(im, None)
        boxes = frcnn_output['boxes'].detach().cpu()
        labels = frcnn_output['labels'].detach().cpu()
        scores = frcnn_output['scores'].detach().cpu()
        
        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            # break
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=1, color=[0, 0, 255])
            label = labels[idx].item()
            label = dataset_test.idxzsd_to_idx[label]
            label = dataset_test.idx2label[label]
            
            text = '{} : {:.2f}'.format(label, scores[idx])

            cv2.putText(gt_im, text=text,
                        org=(x1+5, y1-5),
                        thickness=1,
                        fontScale=1,
                        color=(255, 255, 255),
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        im_name = fname.split('/')[-1]
        cv2.imwrite('./dump/{}'.format(im_name), gt_im)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc_seen.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    args = parser.parse_args()
    if args.infer_samples:
        infer(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')
        
    # if args.evaluate:
    #     evaluate_map(args)
    # else:
    #     print('Not Evaluating as `evaluate` argument is False')