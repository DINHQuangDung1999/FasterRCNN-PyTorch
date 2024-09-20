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
from tools.eval import predict_coco, evaluate_coco
from tools.doc_utils import write_log
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
        semantic_embedding = get_embeddings(dataset_config['dataset_name'], 'w2v', load_background = False)
    # test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    if args.model == 'descreg':
        from model.faster_rcnn_zsd_descreg import FasterRCNN as FasterRCNNDescReg
        
        detector = FasterRCNNDescReg(model_config, 
                                    num_classes=dataset_config['num_classes'],
                                    num_seen_classes=dataset_config['num_seen_classes'], 
                                    semantic_embedding=semantic_embedding)    
    elif args.model == 'baseline':
        from model.faster_rcnn_zsd_baseline import FasterRCNN as FasterRCNNBaseline

        detector = FasterRCNNBaseline(model_config, 
                                    num_classes=dataset_config['num_classes'],
                                    num_seen_classes=dataset_config['num_seen_classes'], 
                                    semantic_embedding=semantic_embedding)      
    elif args.model == 'contrastzsd':
        from model.faster_rcnn_zsd_contrastZSD import FasterRCNN as FasterRCNNContrastZSD

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
    return detector, dataset_test, semantic_embedding

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
    parser.add_argument('--config_path', default='config/xView_trad_resnet.yaml', type=str)
    parser.add_argument('--n_images', default=100, type=int)
    parser.add_argument('--model', default='conventional-conSE', type=str)
    parser.add_argument('--mode', default='unseen', type=str)
    parser.add_argument('--ckpt', default='/home/qdinh/FasterRCNN-PyTorch/checkpoints/xView/frcnn_trad_15-8_10.pt', type=str)

    args = parser.parse_args()
    detector, dataset_test, semantic_embedding = load_model_and_dataset(args)
    detector.eval()
    if args.mode == 'seen':
        predict_coco(detector, dataset_test, style = 'trad')
    elif args.mode == 'unseen':
        predict_coco(detector, dataset_test, style = 'zsd', semantic_embedding = semantic_embedding)
    stats, class_aps = evaluate_coco(dataset_test, draw_PRcurves = False, return_classap=True)    
    epoch_num = int(args.ckpt.split('/')[-1].split('_')[-1].strip('.pt'))
    write_log(dataset_test = dataset_test, epoch_num = epoch_num, stats_unseen = stats, class_aps_unseen = class_aps)
