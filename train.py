import torch
import argparse
import os
import numpy as np
import yaml
import random
from easydict import EasyDict
from model.faster_rcnn import FasterRCNN
from dataset.transforms import get_transforms
from tqdm import tqdm
from dataset.coco import CocoDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tools.eval import predict_coco, evaluate_coco
from tools.doc_utils import write_log
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    args = EasyDict({'config_path': 'config/dior_trad_resnet.yaml',
                     'checkpoint': None,
                    #  'checkpoint': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DOTA/frcnn_trad_15-8_6.pt'
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
                                # transforms=get_transforms(training = True),
                                is_zsd = True)
    # dataset_train[0]
    dataloader_train = DataLoader(dataset_train,
                               batch_size=1,
                               shuffle=False,
                               num_workers=4)
    
    dataset_test = CocoDataset(split = 'test',
                                root_dir=dataset_config['root_dir'],
                                dataset_name = dataset_config['dataset_name'],
                                set_name = dataset_config['testseen_setname'],
                                is_zsd = True)
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if args.checkpoint is not None:
        faster_rcnn_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        start_epoch = int(args.checkpoint.split('_')[-1].strip('.pt')) + 1
        print(f'Continue training from epoch {start_epoch}')
    else: 
        start_epoch = 0 
        print(f'Start training from epoch {start_epoch}')

    optimizer = torch.optim.Adam(faster_rcnn_model.parameters(), lr=train_config['lr'])    
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1
    
    for epoch_num in range(start_epoch, num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        
        for iter_num, data in enumerate(tqdm(dataloader_train)):

        #     break 
        # break 
        
            # try:
            im, target, im_path, im_id = data
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)

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
            # faster_rcnn_model.eval()
            # faster_rcnn_model(im)
            # breakpoint()
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss
            loss = loss / acc_steps

            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())

            if     np.isnan(rpn_output['rpn_classification_loss'].item()) \
                or np.isnan(rpn_output['rpn_localization_loss'].item()) \
                or np.isnan(frcnn_output['frcnn_classification_loss'].item()) \
                or np.isnan(frcnn_output['frcnn_localization_loss'].item()):
                breakpoint()

            if iter_num % 1000 == 0:
                loss_output = 'Epoch {} | Iter {} | '.format(epoch_num, iter_num)
                loss_output += 'RPN Cls Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
                loss_output += ' | RPN Reg : {:.4f}'.format(np.mean(rpn_localization_losses))
                loss_output += ' | FRCNN Cls Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
                loss_output += ' | FRCNN Reg Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
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
        print('Done Training...')
        # os.makedirs(train_config['ckpt_path'], exist_ok=True)
        # checkpoint_dir = os.path.join(train_config['ckpt_path'], train_config['ckpt_name'] + f'_15-8_{epoch_num}.pt')
        # torch.save(faster_rcnn_model.state_dict(), checkpoint_dir)        
        # if epoch_num % train_config['eval_every'] == 0 or epoch_num == num_epochs - 1:
        #     faster_rcnn_model.eval()
        #     print('Evaluating...')
        #     predict_coco(faster_rcnn_model, dataset_test, style = 'trad')
        #     stats, class_aps = evaluate_coco(dataset_test, draw_PRcurves = False, return_classap=True)    
        #     write_log(dataset_test = dataset_test, epoch_num = epoch_num, stats_seen = stats, class_aps_seen = class_aps)
        #     faster_rcnn_model.train()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config_path' , default = 'config/voc_seen.yaml', type=str)
    parser.add_argument('--checkpoint'  , default = '/home/qdinh/FasterRCNN-PyTorch/checkpoints/PascalVOC/frcnn_seen_19.pt',
                        type=str)

    args = parser.parse_args()
    train(args)
