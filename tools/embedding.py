import json
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from scipy.linalg import sqrtm 
from numpy.linalg import inv 


def get_embeddings(dataset_name, type = 'w2v', whitening = False):
    print('Load', type, 'embeddings')
    if type == 'w2v':
        emb_size = 300
    if dataset_name == 'DOTA':
        
        f = open(f'../data/DOTA/{type}_words.json', 'r')
        class_embeddings = json.load(f)
        f = open(f'../data/DOTA/annotations/instances_test2017.json', 'r')
        f = json.load(f)
        class_map = {x['id']: x['name'] for x in f['categories']}

        seen_ids = [1,2,3,4,6,7,8,9,10,11,12,16]
        unseen_ids = [5,13,14,15]

    elif dataset_name == 'DIOR':
        f = open(f'../data/DIOR/{type}_words.json', 'r')
        class_embeddings = json.load(f)
        f = open(f'../data/DIOR/annotations/instances_trainval.json', 'r')
        f = json.load(f)
        class_map = {x['id']: x['name'] for x in f['categories']}

        seen_ids = [1,3,5,6,7,8,9,10,12,13,14,15,16,17,18,19]
        unseen_ids = [2,4,11,20]

    elif dataset_name == 'PascalVOC':
        if type == 'att':
            class_embeddings = np.loadtxt('../data/PascalVOC/VOC/VOC_att.txt', dtype=np.float32, delimiter = ',').T
            return class_embeddings
        else:
            f = open(f'../data/PascalVOC/{type}_words.json', 'r')
            class_embeddings = json.load(f)
        f = open(f'../data/PascalVOC/annotations/instances_test2007.json', 'r')
        f = json.load(f)
        class_map = {x['id']: x['name'] for x in f['categories']}

        seen_ids = [1,2,3,4,5,6,8,9,10,11,13,14,15,16,17,20]
        unseen_ids = [7,12,18,19]

    elif dataset_name == 'MSCOCO':

        f = open(f'../data/MSCOCO/{type}_words.json', 'r')
        class_embeddings = json.load(f)
        f = open(f'../data/MSCOCO/annotations/instances_val2014seenmini.json', 'r')
        f = json.load(f)
        class_map = {x['id']: x['name'] for x in f['categories']}        

        seen_ids = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18, 19, 20, 21, 22, 24, 25, 27, 28, 
                            31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 
                            57, 59, 60, 61, 62, 63, 64, 65, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
        unseen_ids = [5, 7, 14, 17, 23, 33, 34, 36, 48, 54, 58, 70, 74, 80, 89]

    tmp = []
    tmp.append([0]*emb_size)
    for i in seen_ids:
        tmp.append(class_embeddings[class_map[i]])
    for i in unseen_ids:
        tmp.append(class_embeddings[class_map[i]])
    class_embeddings = np.array(tmp)
    # whitening 
    if whitening == True:
        unseen_embs = class_embeddings[len(seen_ids):,:]
        scaler = StandardScaler()
        unseen_embs_centered = scaler.fit_transform(unseen_embs.T).T
        Sigma_centered = unseen_embs_centered @ unseen_embs_centered.T/(unseen_embs_centered.shape[0]-1)
        Sigma_inv = inv(Sigma_centered)
        Sigma_inv_half = sqrtm(Sigma_inv)
        unseen_embs_zca = Sigma_inv_half @ unseen_embs_centered 
        class_embeddings[len(seen_ids):,:] = unseen_embs_zca
    # standardize 
    scaler = StandardScaler()
    class_embeddings = scaler.fit_transform(class_embeddings.T).T
    # normalize
    class_embeddings = normalize(class_embeddings, 'l2', axis = 1)
    return class_embeddings