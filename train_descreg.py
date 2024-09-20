import torch
import argparse
import os
os.chdir('FasterRCNN-PyTorch')
import numpy as np
import yaml
import random
from easydict import EasyDict
from model.faster_rcnn_zsd_descreg import FasterRCNN
from tqdm import tqdm
from dataset.coco import CocoDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tools.embedding import get_embeddings
from tools.eval_descreg import predict_coco, evaluate_coco
from tools.doc_utils import write_log
from visualize import infer
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    args = EasyDict({'config_path': 'config/dior_zsd_resnet_descreg.yaml',
                     'checkpoint': None,
                    #  'checkpoint': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_descreg_1.pt',
                     'pretrained_path': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_trad_15-8_20.pt'
                     })
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

    dataset_train = CocoDataset(split = 'train',
                                root_dir=dataset_config['root_dir'],
                                dataset_name = dataset_config['dataset_name'],
                                set_name = dataset_config['train_setname'],
                                is_zsd = True)
    dataloader_train = DataLoader(dataset_train,
                               batch_size=1,
                               shuffle=False,
                               num_workers=4)
    dataset_testseen = CocoDataset(split = 'test',
                                root_dir=dataset_config['root_dir'],
                                dataset_name = dataset_config['dataset_name'],
                                set_name = dataset_config['testseen_setname'],
                                is_zsd = True)
    dataset_testunseen = CocoDataset(split = 'test',
                                root_dir=dataset_config['root_dir'],
                                dataset_name = dataset_config['dataset_name'],
                                set_name = dataset_config['testunseen_setname'],
                                is_zsd = True)
    if model_config['style'] == 'zsd':
        semantic_embedding = get_embeddings(dataset_config['dataset_name'], 'w2v')
    elif model_config['style'] == 'trad':
        semantic_embedding = None
    
    faster_rcnn_model = FasterRCNN(model_config, 
                                   num_classes=dataset_config['num_classes'],
                                   num_seen_classes=dataset_config['num_seen_classes'], 
                                   semantic_embedding=semantic_embedding)
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if args.checkpoint is not None:
        faster_rcnn_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        start_epoch = int(args.checkpoint.split('_')[-1].strip('.pt')) + 1
        print(f'Continue training from epoch {start_epoch}')
    else: 
        # Load pretrained model 
        pretrained_dict = torch.load(args.pretrained_path, map_location=device)
        model_dict = faster_rcnn_model.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_dict.keys():
                model_dict[k] = v
            else:
                print(k, 'is not in zsd model.')
        faster_rcnn_model.load_state_dict(model_dict)
        print('Load pretrained model successfully')
        start_epoch = 0 
        print(f'Start training from epoch {start_epoch}')
    for k, v in faster_rcnn_model.named_parameters():
        if 'emb_projection' not in k:
            v.requires_grad = False 
    for k, v in faster_rcnn_model.named_parameters():
        if v.requires_grad == True:
            print(f'Trainable layers: {k}')
    # optimizer = torch.optim.SGD(lr=train_config['lr'],
    #                             params=filter(lambda p: p.requires_grad,
    #                                           faster_rcnn_model.parameters()),
    #                             weight_decay=5E-4,
    #                             momentum=0.9)
    optimizer = torch.optim.Adam(faster_rcnn_model.parameters(), lr=train_config['lr']) 
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1
    
    for epoch_num in range(start_epoch, num_epochs):

        frcnn_classification_losses = []
        frcnn_triplet_losses = []

        for iter_num, data in enumerate(tqdm(dataloader_train)):
        #     break 
        # break 
            im, target, im_path, im_id = data
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            # faster_rcnn_model.eval()
            # faster_rcnn_model(im, None, pred_style = 'trad')
            tmp = target['bboxes'].clone()
            zero_width_idx = tmp[:,:,0] - tmp[:,:, 2] == 0
            if  zero_width_idx.sum() > 0:
                print(f'Warning. Encounter 0 width gt_boxes at im: {im_path[0].split("/")[-1]}, box: {tmp[zero_width_idx].cpu()[0]}')
                target['bboxes'] = target['bboxes'][~zero_width_idx].unsqueeze(0)
            zero_height_idx = tmp[:,:,1] - tmp[:,:, 3] == 0
            if  zero_height_idx.sum() > 0:
                print(f'Warning. Encounter 0 height gt_boxes at im: {im_path[0].split("/")[-1]}, box: {tmp[zero_height_idx].cpu()[0]}')
                target['bboxes'] = target['bboxes'][~zero_height_idx].unsqueeze(0)            
            # breakpoint()
            
            optimizer.zero_grad()
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_triplet_loss']

            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_triplet_losses.append(frcnn_output['frcnn_triplet_loss'].item())

            if iter_num % 20 == 0:
                loss_output = 'Epoch {} | Iter {}'.format(epoch_num, iter_num)
                loss_output += ' | FRCNN Cls Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
                loss_output += ' | FRCNN Triplet Loss : {:.4f}'.format(np.mean(frcnn_triplet_losses))
                print(loss_output)


            grad_norm = torch.nn.utils.clip_grad_norm_(faster_rcnn_model.parameters(), 0.001)
            loss.backward()
            
            if step_count % acc_steps == 0:
                optimizer.step()
            step_count += 1

            # except Exception as e:
            #     print(e)
            #     breakpoint() 
        scheduler.step()
        print('Finished epoch {}'.format(epoch_num))
        os.makedirs(train_config['ckpt_path'], exist_ok=True)
        checkpoint_dir = os.path.join(train_config['ckpt_path'], train_config['ckpt_name'] + f'_descreg_{epoch_num}.pt')
        torch.save(faster_rcnn_model.state_dict(), checkpoint_dir)
        print('Done Training...')
        
        # infer images at each epochs
        os.makedirs(f'./dump/DIOR/descreg/seen/epoch{epoch_num}', exist_ok=True)
        infer(faster_rcnn_model.eval(), dataset_testseen, 'trad', 0.6, n_images = 20, dump_dir = f'./dump/DIOR/descreg/seen/epoch{epoch_num}')
        os.makedirs(f'./dump/DIOR/descreg/unseen/epoch{epoch_num}', exist_ok=True)
        infer(faster_rcnn_model.eval(), dataset_testunseen, 'zsd', 0.1, n_images = 20, dump_dir = f'./dump/DIOR/descreg/unseen/epoch{epoch_num}')
        faster_rcnn_model.train()
        # # Evaluation
        # if epoch_num % train_config['eval_every'] == 0 or epoch_num == num_epochs - 1:
        #     faster_rcnn_model.eval()
        #     print('Evaluating...')
        #     predict_coco(faster_rcnn_model, dataset_test, pred_style = 'trad')
        #     stats, class_aps = evaluate_coco(dataset_test, draw_PRcurves = False, return_classap=True)    
        #     write_log(dataset_test = dataset_test, epoch_num = epoch_num, stats_seen = stats, class_aps_seen = class_aps)
        # faster_rcnn_model.train()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config_path' , default = 'config/voc_seen.yaml', type=str)
    parser.add_argument('--checkpoint'  , default = '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DIOR/frcnn_seen_19.pt',
                        type=str)

    args = parser.parse_args()
    train(args)
