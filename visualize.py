import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
import shutil
from easydict import EasyDict
from tqdm import tqdm
from dataset.coco import CocoDataset
from tools.embedding import get_embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_dataset(args):
    # Read the config file #

    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
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
    if args.mode == 'seen':
        dataset_test = CocoDataset(split = 'test',
                                    root_dir=dataset_config['root_dir'],
                                    dataset_name = dataset_config['dataset_name'],
                                    set_name = dataset_config['testseen_setname'],
                                    is_zsd = True)
    elif args.mode == 'unseen':
        dataset_test = CocoDataset(split = 'test',
                                    root_dir=dataset_config['root_dir'],
                                    dataset_name = dataset_config['dataset_name'],
                                    set_name = dataset_config['testunseen_setname'],
                                    is_zsd = True)
    # test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    if args.model == 'descreg':
        from model.faster_rcnn_zsd_descreg import FasterRCNN as FasterRCNNDescReg
        semantic_embedding = get_embeddings(dataset_config['dataset_name'], 'w2v')
        detector = FasterRCNNDescReg(model_config, 
                                    num_classes=dataset_config['num_classes'],
                                    num_seen_classes=dataset_config['num_seen_classes'], 
                                    semantic_embedding=semantic_embedding)    
    elif args.model == 'baseline':
        from model.faster_rcnn_zsd_baseline import FasterRCNN as FasterRCNNBaseline
        semantic_embedding = get_embeddings(dataset_config['dataset_name'], 'w2v')
        detector = FasterRCNNBaseline(model_config, 
                                    num_classes=dataset_config['num_classes'],
                                    num_seen_classes=dataset_config['num_seen_classes'], 
                                    semantic_embedding=semantic_embedding)      
    elif args.model == 'contrastzsd':
        from model.faster_rcnn_zsd_contrastZSD import FasterRCNN as FasterRCNNContrastZSD
        semantic_embedding = get_embeddings(dataset_config['dataset_name'], 'w2v')
        detector = FasterRCNNContrastZSD(model_config, 
                                    num_classes=dataset_config['num_classes'],
                                    num_seen_classes=dataset_config['num_seen_classes'], 
                                    semantic_embedding=semantic_embedding)             
    else:
        from model.faster_rcnn import FasterRCNN
        detector = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    detector.eval()
    detector.to(device)
    detector.load_state_dict(torch.load(os.path.join(train_config['ckpt_path'],args.ckpt),map_location=device))
    return detector, dataset_test

def infer(detector, dataset_test, style, low_score_threshold = 0.05, 
          put_gt_label = False, n_images = 100, dump_dir = './dump'):
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)

    # Hard coding the low score threshold for inference on images for now
    # Should come from config
    detector.roi_head.low_score_threshold = low_score_threshold
    if style == 'zsd':
        semantic_embedding = get_embeddings(dataset_test.dataset_name, 'w2v', load_background= False)
        # semantic_embedding = semantic_embedding[1:,:] # exclude background
    elif style == 'trad':
        semantic_embedding = None
    # im_idx = [13891, 13985, 14020, 14027, 14028, 15977, 16264, 16282, 18455, 18929, 20041, 17115, 19758, 21230, 21435, 21780, 21789, 22342]        
    # for sample_count in tqdm(range(n_images)):
    # for sample_count in tqdm(im_idx):
    for sample_count in tqdm(range(len(dataset_test))):
        random_idx = random.randint(0, len(dataset_test) - 1)
        # random_idx = sample_count
        im, target, fname, im_id = dataset_test[random_idx]
        # breakpoint()
        # if im_id not in im_idx:
        #     continue
        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)

        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = target['labels'][idx].detach().cpu().item()
            label = dataset_test.idxzsd2idx[label]
            label = dataset_test.idx2label[label]
            text = label
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=1, color=[0, 255, 0])
            if put_gt_label == True:
                cv2.putText(gt_im, text=text,
                            org=(x1+5, y1+15),
                            thickness=1,
                            fontScale=1,
                            color=(255, 255, 255),
                            fontFace=cv2.FONT_HERSHEY_PLAIN)
    
        # Getting predictions from trained model
        try:
            rpn_output, frcnn_output = detector(im, None, style, semantic_embedding)
        except:
            rpn_output, frcnn_output = detector(im, None, style)
        # rpn_output, frcnn_output = detector(im, None, style)
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
            label = dataset_test.idxzsd2idx[label]
            label = dataset_test.idx2label[label]
            
            text = '{} : {:.2f}'.format(label, scores[idx])

            cv2.putText(gt_im, text=text,
                        org=(x1, y1+20),
                        thickness=1,
                        fontScale=1,
                        color=(255, 255, 255),
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        im_name = fname.split('/')[-1]
        cv2.imwrite(f'{dump_dir}/{im_name}', gt_im)

# args = EasyDict({'config_path':'config/dior_trad_resnet.yaml',
#                  'model': 'conventional-conSE',
#                  'type':'unseen',
#                  'ckpt': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_trad_15-8_20.pt'})


# args = EasyDict({'config_path': 'config/dior_zsd_resnet_baseline.yaml',
#                  'model': 'baseline',
#                  'type':'seen',
#                  'ckpt': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_baseline_0.pt',
#                     })

# args = EasyDict({'config_path': 'config/dior_zsd_resnet_descreg.yaml',
#                  'model': 'descreg',
#                  'type':'unseen',
#                  'ckpt': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_descreg_0.pt',
#                     })

# args = EasyDict({'config_path': 'config/dior_zsd_resnet_contrastzsd.yaml',
#                  'model': 'contrastzsd',
#                  'type':'seen',
#                  'ckpt': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_contrastzsd_10.pt',
#                     })

# style = 'zsd'
# detector, dataset_test = load_model_and_dataset(args)
# infer(detector, dataset_test, style, 0.1, put_gt_label=False, dump_dir = f'./dump/DIOR/{args.model}/{args.mode}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    parser.add_argument('--config_path', default='config/dior_trad_resnet.yaml', type=str)
    parser.add_argument('--n_images', default=100, type=int)
    parser.add_argument('--model', default='conventional-conSE', type=str)
    parser.add_argument('--mode', default='unseen', type=str)
    parser.add_argument('--ckpt', default='/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_trad_15-8_20.pt', type=str)

    args = parser.parse_args()
    detector, dataset_test = load_model_and_dataset(args)
    if args.mode == 'seen':
        infer(detector, dataset_test, 'trad', 0.1, put_gt_label=False, dump_dir = f'./dump/DIOR/{args.model}/{args.mode}')
    elif args.mode == 'unseen':
        infer(detector, dataset_test, 'zsd', 0.15, put_gt_label=False, dump_dir = f'./dump/DIOR/{args.model}/{args.mode}')