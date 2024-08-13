import torch
import argparse
import os
os.chdir('FasterRCNN-PyTorch')
import numpy as np
import yaml
import random
from easydict import EasyDict
from model.faster_rcnn import FasterRCNN
from tqdm import tqdm
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tools.embedding import get_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    args = EasyDict({'config_path': 'config/voc_zsd.yaml',
                     'pretrained': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/PascalVOC/frcnn_trad_11.pt',
                     'checkpoint': None})
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
    
    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     is_zsd = True)
    dataloader_train = DataLoader(voc,
                               batch_size=1,
                               shuffle=False,
                               num_workers=4)
    
    semantic_embedding = get_embeddings('PascalVOC')
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'], semantic_embedding=semantic_embedding)
    # Load pretrained weight
    pretrained_dict = torch.load(args.pretrained)
    model_dict = faster_rcnn_model.state_dict()
    for k, v in pretrained_dict.items():
        if k in model_dict:
            model_dict[k] = pretrained_dict[k]
        else:
            print(k, 'not found in model params')
    faster_rcnn_model.load_state_dict(model_dict)
    # Second training stage: Train only the projection with triplet loss
    for k, v in faster_rcnn_model.named_parameters():
        if 'feat_projection' not in k:
            v.requires_grad = False 
        if v.requires_grad == True:
            print(k)
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    # #####
    # im, target, fname = next(iter(dataloader_train))
    # im = im.cuda()
    # target['bboxes'] = target['bboxes'].cuda()
    # target['labels'] = target['labels'].cuda()
    # faster_rcnn_model(im, target)
    # #####
    if args.checkpoint is not None:
        faster_rcnn_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        start_epoch = int(args.checkpoint.split('_')[-1].strip('.pt'))
        print(f'Continue training from epoch {start_epoch}')
    else: 
        start_epoch = 0 
        print(f'Start training from epoch {start_epoch}')
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
    # breakpoint()
    for epoch_num in range(start_epoch, num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []

        for iter_num, data in enumerate(tqdm(dataloader_train)):
        #     break
        # break
            im, target, _ = data
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)

            optimizer.zero_grad()

            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            # breakpoint()
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss
            loss = loss / acc_steps
            loss.backward()

            if step_count % acc_steps == 0:
                optimizer.step()
                
            step_count += 1
            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())

            loss_output = 'Iter {} | '.format(iter_num)
            loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
            loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
            loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
            loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
            print(loss_output)
        scheduler.step()
        print('Finished epoch {}'.format(epoch_num))
        os.makedirs(train_config['ckpt_path'], exist_ok=True)
        checkpoint_dir = os.path.join(train_config['ckpt_path'], train_config['ckpt_name'] + f'_{epoch_num}.pt')
        torch.save(faster_rcnn_model.state_dict(), checkpoint_dir)
    print('Done Training...')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config_path' , default = 'config/voc_seen.yaml', type=str)
    parser.add_argument('--checkpoint'  , default = '/home/qdinh/FasterRCNN-PyTorch/checkpoints/PascalVOC/frcnn_seen_19.pt',
                        type=str)

    args = parser.parse_args()
    train(args)
