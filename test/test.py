# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random
import math
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import glob
import torch
import torch.nn as nn
import torchvision.transforms as ttransforms

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain_PL
from nets.yolo_loss import YOLOLoss_hrnet
from common.coco_dataset_bp_yolo import COCODataset
from common.evaluation import accuracy, AverageMeter, final_preds


unloader = ttransforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image
def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds
def test(config):

    f_log=open(config["pretrain_floder"].replace('*.pth','')+"log.log",'w')
    is_training = False
    slice_num=config["yolo"]["line_num"]
    # Load and initialize network
    net = ModelMain_PL(config, is_training=is_training)
    net.train(is_training)
    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()
    dis=0
    idx = [1]
    dataloader = torch.utils.data.DataLoader(COCODataset(config["test_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=False),
                                             batch_size=config["batch_size"],
                                             shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    pretrain_files = sorted(glob.glob(config["pretrain_floder"]))
    loss_coor = YOLOLoss_hrnet(config["yolo"]["anchors"][0], config["lambda_xy"], (config["img_w"], config["img_h"]))
    for files in pretrain_files:
        acces = AverageMeter()
        acces_h = AverageMeter()
        Drms1=[]
        Drms2=[]
        Drms=[]
        err_list=[]
        if files:
            logging.info("load checkpoint from {}".format(files))
            state_dict = torch.load(files)
            net.load_state_dict(state_dict)
        else:
            raise Exception("missing pretrain_snapshot!!!")

        for step, samples in enumerate(dataloader):
            images, labels = samples["image"], samples["label"]
            # labels_h = samples["label_h"]
            labels = labels.cuda()
            # labels_h = labels_h.cuda()
            labels_coor = samples["label_coor"]
            path = samples["image_path"]
            # labels_coor = labels_coor.cuda()
            with torch.no_grad():
                pred, predh, coord = net(images)
                det = loss_coor(coord)

                for id, detections in enumerate(det):
                # for id, detections in enumerate(detl):
                    _,indextensor=detections.max(0)
                    detections=detections[indextensor[2]].unsqueeze(0)

                    for x1, y1, conf in detections:
                        ori_h, ori_w = images[id].shape[1:3]
                        org_x = labels_coor[id][0][1]*config["img_h"]
                        org_y = labels_coor[id][0][2]*config["img_h"]
                        # print(x1,y1,org_x,org_y)
                        err = math.sqrt(pow(float(org_x - x1),2)+pow(float(org_y - y1),2))
                        err = err / (256.*math.sqrt(2))
                        if err>0.1:
                            img = cv2.imread(path[id],cv2.IMREAD_UNCHANGED)
                            cv2.imwrite(path[id].replace('/home/lyb/datasets/vp/val/kong','/hard/result/need'),img)
                            err=0.1

                        err_list.append(err)
                        img = cv2.imread(path[id],cv2.IMREAD_UNCHANGED)
                        img = cv2.resize(img, (config["img_h"],config["img_h"]), interpolation=cv2.INTER_CUBIC)
                        img = cv2.circle(img,(int(org_x),int(org_y)),3,(255,0,0),-1)
                        img = cv2.circle(img,(x1,y1),3,(0,0,255),-1)
                        cv2.imwrite(path[id].replace('/home/lyb/datasets/vp/val','/hard/result'),img)

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python test_images.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    for i in config["pretrain_floder_list"]:
        config["pretrain_floder"]=i
        test(config)


if __name__ == "__main__":
    main()
