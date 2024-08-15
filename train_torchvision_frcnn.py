import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.coco import CocoDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

from easydict import EasyDict
import torch.nn as nn 
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_function(data):
    return tuple(zip(*data))

class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _EuDist(a, b):
        '''
            Expect: a, b 1D tensor
        '''
        return torch.sqrt(torch.sum((a - b)**2))

    def _seSoftmax(tensor):
        '''
            Expect 2D-tensor
            Return: 2D-tensor
        '''
        _tensor = tensor.clone()
        for row in range(tensor.shape[0]):
            index = _tensor[row,:] != 1
            _tensor[row, index] = F.softmax(_tensor[row, index], dim = -1)
        return _tensor

    def forward(self, semantic_embeddings, projected_embeddings):
        '''
            Args:
            **w**: projected_class_embeddings
        '''
        semantic_embeddings = torch.from_numpy(semantic_embeddings).float().to(device)
        semantic_embeddings = torch.nn.functional.normalize(semantic_embeddings, p = 2, dim = -1)
        S = semantic_embeddings @ semantic_embeddings.T
        S = self._seSoftmax(S)

        n_classes = S.shape[0]

        L = []
        for j in range(n_classes):
            sorted_idx = torch.argsort(-S[j,:])
            most_similar = sorted_idx[1].item()
            least_similar = sorted_idx[-1].item()
            margin = S[j, most_similar] - S[j, least_similar]

            d_similar = self._EuDist(projected_embeddings[j,:], projected_embeddings[most_similar,:])
            d_dissimilar = self._EuDist(projected_embeddings[j,:], projected_embeddings[least_similar, :])
            loss = torch.max(torch.tensor(0.).to(device), 
                             d_similar - d_dissimilar + margin)

            L.append(loss)
            # breakpoint()

        L = torch.stack(L)
        L = torch.mean(L)
        
        return L
    

class ZSDPredictor(nn.Module):
    """
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes, semantic_embedding, emb_dim = None):
        super().__init__()
        # self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
        self.semantic_embedding = semantic_embedding

        self.triplet_loss = TripletLoss()
        # normalized_emb = torch.nn.functional.normalize(self.semantic_embedding, p = 2, dim = -1)
        # self.cosine_embedding = normalized_emb @ normalized_emb.T
        # self.cosine_embedding = seSoftmax(self.cosine_embedding)
        if emb_dim is None:
            self.emb_dim = semantic_embedding.shape[1]
        else:
            self.emb_dim = emb_dim

        self.feat_projection = nn.Linear(self.fc_inner_dim, self.emb_dim, bias = True)
        # self.emb_projection = nn.Linear(semantic_embedding.shape[1], self.emb_dim, bias = True)
        # self.emb_projection.weight.data.copy_(torch.eye(self.emb_dim, requires_grad=False))

        self.emb_projection = nn.Linear(semantic_embedding.shape[1], self.emb_dim, bias = True)

    def forward(self, x):
        # if x.dim() == 4:
        #     torch._assert(
        #         list(x.shape[2:]) == [1, 1],
        #         f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
        #     )
        # x = x.flatten(start_dim=1)
        # scores = self.cls_score(x)

        projected_visual_feats = self.feat_projection(x)
        projected_emb_feats = self.emb_projection(self.semantic_embedding.float())
        normed_projected_visual_feats = torch.nn.functional.normalize(projected_visual_feats, dim = -1)
        normed_projected_emb_feats = torch.nn.functional.normalize(projected_emb_feats, dim = -1)
        scores = normed_projected_visual_feats @ normed_projected_emb_feats.T

        bbox_deltas = self.bbox_pred(x)

        trip_loss = self.triplet_loss(self.semantic_embedding, projected_emb_feats)
        classification_loss += trip_loss
        return scores, bbox_deltas, trip_loss
    
def train(args):
    args = EasyDict({'config_path': 'config/dior_trad_resnet.yaml',
                     'checkpoint': None,
                    #  'checkpoint': '/home/qdinh/FasterRCNN-PyTorch/checkpoints/DOTA/frcnn_trad_0.pt',
                     'use_resnet50_fpn': True,
                     })
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    
    dataset_config = config['dataset_params']
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
                               batch_size=4,
                               shuffle=False,
                               num_workers=4,
                               collate_fn = collate_function)

    if args.use_resnet50_fpn:

        rpn_anchor_generator = AnchorGenerator(((16,), (32,), (64,), (128,), (256,)), ((0.5, 1.0, 2.0),) * 5)
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                                 min_size=600,
                                                                                 max_size=1000,
                                                                                 rpn_anchor_generator = rpn_anchor_generator,
        )
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
            faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=dataset_config['num_classes'])
    else:
        backbone = torchvision.models.resnet34(pretrained=True, norm_layer=torchvision.ops.FrozenBatchNorm2d)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = 256
        roi_align = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        rpn_anchor_generator = AnchorGenerator()
        faster_rcnn_model = torchvision.models.detection.FasterRCNN(backbone,
                                                                    num_classes=21,
                                                                    min_size=600,
                                                                    max_size=1000,
                                                                    rpn_anchor_generator=rpn_anchor_generator,
                                                                    rpn_pre_nms_top_n_train=12000,
                                                                    rpn_pre_nms_top_n_test=6000,
                                                                    box_batch_size_per_image=128,
                                                                    rpn_post_nms_top_n_test=300
                                                                    )

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config['ckpt_path']):
        os.mkdir(train_config['ckpt_path'])

    optimizer = torch.optim.SGD(lr=1E-4,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5, momentum=0.9)

    num_epochs = train_config['num_epochs']
    step_count = 0

    for epoch_num in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        for iter_num, data in enumerate(tqdm(dataloader_train)): 
            ims, targets, fname, im_id = data
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']
            # print(batch_losses['loss_classifier'], batch_losses['loss_box_reg'], batch_losses['loss_rpn_box_reg'], batch_losses['loss_objectness'])
            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            if iter_num % 50 == 0:
                loss_output = 'Epoch {} | Iter {} | '.format(epoch_num, iter_num)
                loss_output += 'RPN Cls Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
                loss_output += ' | RPN Reg : {:.4f}'.format(np.mean(rpn_localization_losses))
                loss_output += ' | FRCNN Cls Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
                loss_output += ' | FRCNN Reg Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
                print(loss_output)

            loss.backward()
            optimizer.step()
            step_count +=1

        print('Finished epoch {}'.format(i))
        if args.use_resnet50_fpn:
            torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['ckpt_path'], f'tv_frcnn_r50fpn_{epoch_num}'))
        else:
            torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['ckpt_path'], f'tv_frcnn_{epoch_num}'))
        # loss_output = ''
        # loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        # loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        # loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        # loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        # print(loss_output)
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    args = parser.parse_args()
    train(args)
