import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seSoftmax(tensor):
    '''
        Expect 2D-tensor
        Return: 2D-tensor
    '''
    _tensor = tensor.clone()
    for row in range(tensor.shape[0]):
        index = _tensor[row,:] != 1
        _tensor[row, index] = F.softmax(_tensor[row, index], dim = -1)
    return _tensor

def EuDist(a, b):
    '''
        Expect: a, b 1D tensor
    '''
    return torch.sqrt(torch.sum((a - b)**2))

class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt_class_embeddings, w):
        '''
            Args:
            **w**: projected_class_embeddings
        '''
        gt_class_embeddings = torch.from_numpy(gt_class_embeddings).float().to(device)
        gt_class_embeddings = torch.nn.functional.normalize(gt_class_embeddings, p = 2, dim = -1)
        S = gt_class_embeddings @ gt_class_embeddings.T
        S = seSoftmax(S)

        n_classes = S.shape[0]

        L = []
        for j in range(n_classes):
            sorted_idx = torch.argsort(-S[j,:])
            most_similar = sorted_idx[1].item()
            least_similar = sorted_idx[-1].item()
            margin = S[j, most_similar] - S[j, least_similar]
    
            loss = torch.max(torch.tensor(0.).to(device), EuDist(w[j,:], w[most_similar,:]) - EuDist(w[j,:], w[least_similar, :]) + margin)

            L.append(loss)
            # breakpoint()

        L = torch.stack(L)
        L = torch.mean(L)
        
        return L