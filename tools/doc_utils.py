import datetime
import numpy as np 
import pandas as pd 
import os 

def write_log(dataset_test, epoch_num, 
              stats_seen = None, class_aps_seen = None, 
              stats_unseen = None, class_aps_unseen = None):
    
    os.makedirs(f'log/{dataset_test.dataset_name}', exist_ok=True)
    date = str(datetime.datetime.date(datetime.datetime.today()))
        
    if dataset_test.dataset_name == 'PascalVOC':
        unseen_classes = ['car', 'dog', 'sofa', 'train']
    elif dataset_test.dataset_name == 'DOTA':
        unseen_classes = ['tennis-court', 'helicopter', 'soccer-ball-field', 'swimming-pool']
    elif dataset_test.dataset_name == 'DIOR':
        unseen_classes = ['airport', 'basketballcourt', 'groundtrackfield', 'windmill']
    elif dataset_test.dataset_name == 'xView':
        unseen_classes = ['Helicopter', 'Bus', 'Pickup Truck', 'Truck Tractor w/ Box Trailer', 'Maritime Vessel', \
                'Motorboat', 'Barge', 'Reach Stacker', 'Mobile Crane', 'Scraper/Tractor', 'Excavator', 'Shipping container lot']
    

    if ((stats_seen is not None) and (class_aps_seen is not None)) and ((stats_unseen is None) and (class_aps_unseen is None)):
        seen_classes = [x for x in class_aps_seen.keys() if x not in unseen_classes]
        lines_mAP_seen= [
            f'Date: {date} | Dataset: {dataset_test.dataset_name} | Valset: {dataset_test.set_name} | Epoch {epoch_num} | DetectType: traditional \n'
            f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_seen[0]}\n'
            f'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats_seen[1]}\n'
            f'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats_seen[2]}\n'
            f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_seen[8]}\n'
        ]

        f = open(f'log/{dataset_test.dataset_name}/traditional_mAPs.txt', 'a')
        f.writelines(lines_mAP_seen)
        f.close()

        columns = ['Date', 'Dataset', 'SetName', 'Epoch',\
                   'mAP@0.5', 'R100@0.5'] + seen_classes

        if os.path.exists(f'log/{dataset_test.dataset_name}/traditional_classAPs.csv'):
            f = pd.read_csv(f'log/{dataset_test.dataset_name}/traditional_classAPs.csv')
        else:
            f = pd.DataFrame(columns = columns, index=None)

        line_class_APs = [date, dataset_test.dataset_name, dataset_test.set_name, epoch_num]
        line_class_APs += [stats_seen[1], stats_seen[8]]
        seen_aps = [np.round(v, 3) for (k, v) in class_aps_seen.items() if k in seen_classes]
        line_class_APs += seen_aps

        # unseen_aps = [np.round(v, 3) for (k, v) in class_aps_seen.items() if k in unseen_classes]
        # line_class_APs += unseen_aps

        f = pd.concat([f, pd.DataFrame([line_class_APs], columns=columns)])
        f.to_csv(f'log/{dataset_test.dataset_name}/traditional_classAPs.csv', index = None)

    elif ((stats_seen is None) and (class_aps_seen is None)) and ((stats_unseen is not None) and (class_aps_unseen is not None)):
        unseen_classes = [x for x in class_aps_unseen.keys() if x not in unseen_classes]
        lines_mAP_unseen= [
            f'Date: {date} | Dataset: {dataset_test.dataset_name} | Valset: {dataset_test.set_name} | Epoch {epoch_num} | DetectType: zsd \n'
            f'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_unseen[0]}\n'
            f'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats_unseen[1]}\n'
            f'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats_unseen[2]}\n'
            f'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats_unseen[8]}\n'
        ]

        f = open(f'log/{dataset_test.dataset_name}/unseen_mAPs.txt', 'a')
        f.writelines(lines_mAP_unseen)
        f.close()

        columns = ['Date', 'Dataset', 'SetName', 'Epoch',\
                   'mAP@0.5', 'R100@0.5'] + unseen_classes

        if os.path.exists(f'log/{dataset_test.dataset_name}/unseen_classAPs.csv'):
            f = pd.read_csv(f'log/{dataset_test.dataset_name}/unseen_classAPs.csv')
        else:
            f = pd.DataFrame(columns = columns, index=None)

        line_class_APs = [date, dataset_test.dataset_name, dataset_test.set_name, epoch_num]
        line_class_APs += [stats_unseen[1], stats_unseen[8]]
        unseen_aps = [np.round(v, 3) for (k, v) in class_aps_unseen.items() if k in unseen_classes]
        line_class_APs += unseen_aps

        # unseen_aps = [np.round(v, 3) for (k, v) in class_aps_seen.items() if k in unseen_classes]
        # line_class_APs += unseen_aps

        f = pd.concat([f, pd.DataFrame([line_class_APs], columns=columns)])
        f.to_csv(f'log/{dataset_test.dataset_name}/unseen_classAPs.csv', index = None)