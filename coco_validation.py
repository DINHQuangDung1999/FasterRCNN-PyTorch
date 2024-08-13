import os
os.chdir('./FasterRCNN-PyTorch')
import argparse
from easydict import EasyDict
from tools.eval import load_model_and_dataset, predict_coco, evaluate_coco

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    parser.add_argument('--config_path' , default='config/voc_seen.yaml', type=str)
    parser.add_argument('--ckpt'        , default='frcnn_seen_19.pt', type=str)
    parser.add_argument('--detect'      , default=True, type=bool)
    parser.add_argument('--evalmAP'     , default=True, type=bool)
    parser.add_argument('--drawPRcurves', default=True, type=bool)
    args = parser.parse_args()

    args = EasyDict({   'config_path'   : 'config/voc_trad.yaml',
                        'ckpt'          : 'frcnn_seen_19.pt',
                        'detect'        : False,
                        'evalmAP'       : True,
                        'drawPRcurves'  : False,
                        'write_results' : True})

    detector, dataset_test = load_model_and_dataset(args)

    if args.detect == True:
        predict_coco(detector, dataset_test)
    else:
        print('Not performing detections as `detect` argument is False')

    if args.evalmAP == True:
        stats, class_aps = evaluate_coco(dataset_test, draw_PRcurves = args.drawPRcurves, return_classap=True)
    else:
        print('Not evaluating mAP as `detect` argument is False')

    # if args.write_results == True:
    #     stats, class_aps = evaluate_coco(dataset_test, draw_PRcurves = args.drawPRcurves, return_classap=True)
    # else:
    #     print('Not saving evaluation log as `write_results` argument is False')

