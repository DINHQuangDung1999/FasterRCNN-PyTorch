import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize, StandardScaler
from scipy.linalg import sqrtm 
from numpy.linalg import inv 
import yaml

def get_embeddings(dataset_name, type = 'w2v', load_background = False):
    '''obtained a normalized word embeddings'''
    if type == 'w2v':
        print('Load', type, 'embeddings')
        if dataset_name == 'DOTA':
            set_name = 'test2017'
        elif dataset_name == 'DIOR':
            set_name = 'trainval'
        elif dataset_name == 'xView':
            set_name = 'xView'
        js = open(f'/home/qdinh/data/{dataset_name}/w2v_words.json', 'r')
        class_embeddings = json.load(js)

        f = open(f'/home/qdinh/data/{dataset_name}/annotations/instances_{set_name}.json', 'r')
        f = json.load(f)

        splits = open(f'/home/qdinh/data/{dataset_name}/split.txt')
        splits = splits.readlines()
        seen, unseen = splits 
        seen = seen.strip('\n').split(',')
        unseen = unseen.strip('\n').split(',')

        tmp = []
        for cat in seen:
            tmp.append(class_embeddings[cat])
        for cat in unseen:
            tmp.append(class_embeddings[cat])

        tmp = np.array(tmp)
        if load_background == True:
            emb_size = tmp.shape[1]
            background_ = np.mean(np.array(tmp), axis = 0).reshape(-1, emb_size)
            class_embeddings = np.concatenate([background_, tmp]) # Background vector = mean of other vectors
        else:
            class_embeddings = tmp
        # normalize
        class_embeddings = normalize(class_embeddings, 'l2', axis = 1)
        return class_embeddings
    elif type == 'bert':

        # Read the config file 
        with open(f'/home/qdinh/data/{dataset_name}/GPTdescription.yaml', 'r') as file:
            try:
                GPTdescription = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        splits = open(f'/home/qdinh/data/{dataset_name}/split.txt')
        splits = splits.readlines()
        seen, unseen = splits 
        seen = seen.strip('\n').split(',')
        unseen = unseen.strip('\n').split(',')
        seen_desc = [GPTdescription[key] for key in seen]
        unseen_desc = [GPTdescription[key] for key in unseen]

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        seen_emb = model.encode(seen_desc)
        unseen_emb = model.encode(unseen_desc)

        emb = np.concatenate([seen_emb, unseen_emb])
        mean_vector = np.mean(emb, axis = 0).reshape(1,-1)
        mean_vector = mean_vector / np.sqrt(np.sum(mean_vector**2))
        emb = np.concatenate([mean_vector, emb])

        return emb 

