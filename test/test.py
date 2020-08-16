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

# import torchvision.transforms as ttransforms

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
    # if config["yolo"]["batchnorm"]=="SYBN":
    #     patch_replication_callback(net)
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
                # pred = net(images)
                # pred, coord = net(images)

                pred, predh, coord = net(images)
                # print(coord.shape)
                # for coord
                det = loss_coor(coord)
                # detl=get_preds(pred/255)
                # deth=get_preds(predh/255)
                # print(detl,deth)
                # print(len(det))

                for id, detections in enumerate(det):
                # for id, detections in enumerate(detl):
                    _,indextensor=detections.max(0)
                    # print(indextensor)
                    # print(indextensor.shape)
                    detections=detections[indextensor[2]].unsqueeze(0)
                    # print(detections.shape)

                    for x1, y1, conf in detections:
                    # for x1, y1 in detections:
                        # print(x1, y1,conf)
                        ori_h, ori_w = images[id].shape[1:3]
                        # print(ori_h,ori_w)
                        # print(float(labels_coor[id][1][1])*ori_w-x1)
                        # pre_h, pre_w = config["img_h"], config["img_w"]
                        # y1 = (y1 / pre_h) * ori_h
                        # x1 = (x1 / pre_w) * ori_w

                        org_x = labels_coor[id][0][1]*config["img_h"]
                        org_y = labels_coor[id][0][2]*config["img_h"]
                        # print(x1,y1,org_x,org_y)
                        err = math.sqrt(pow(float(org_x - x1),2)+pow(float(org_y - y1),2))
                        # if org_y<5 or org_x<3:
                        #     print(org_x,org_y,labels_coor[id][0][1],labels_coor[id][0][2],config["img_h"],print(path[id]))
                        # if err >10:
                        #     print(path[id])
                        #     print(org_x,org_y)
                        #     print(x1,y1)
                        err = err / (256.*math.sqrt(2))
                        if err>0.1:
                            # print(path[id])
                            # print(org_x,org_y,labels_coor[id][0][1],labels_coor[id][0][2],config["img_h"],x1,y1)
                            img = cv2.imread(path[id],cv2.IMREAD_UNCHANGED)
                            cv2.imwrite(path[id].replace('/home/lyb/datasets/vp/val/kong','/hard/result/need'),img)
                            err=0.1

                        err_list.append(err)
                        img = cv2.imread(path[id],cv2.IMREAD_UNCHANGED)
                        img = cv2.resize(img, (config["img_h"],config["img_h"]), interpolation=cv2.INTER_CUBIC)
                        img = cv2.circle(img,(int(org_x),int(org_y)),3,(255,0,0),-1)
                        img = cv2.circle(img,(x1,y1),3,(0,0,255),-1)
                        cv2.imwrite(path[id].replace('/home/lyb/datasets/vp/val','/hard/result'),img)
                        # pimg = tensor_to_PIL(pred/255)
                        # pimg.save(path[id].replace('/home/lyb/datasets/vp/val/kong','/hard/result/hmp'))


            # print(path)
        print(len(err_list))
        print(0.01,sum(i<=0.01 for i in err_list),sum(i<=0.01 for i in err_list)/len(err_list))
        print(0.02,sum(i<=0.02 for i in err_list),sum(i<=0.02 for i in err_list)/len(err_list))
        print(0.03,sum(i<=0.03 for i in err_list),sum(i<=0.03 for i in err_list)/len(err_list))
        print(0.04,sum(i<=0.04 for i in err_list),sum(i<=0.04 for i in err_list)/len(err_list))
        print(0.05,sum(i<=0.05 for i in err_list),sum(i<=0.05 for i in err_list)/len(err_list))
        print(0.06,sum(i<=0.06 for i in err_list),sum(i<=0.06 for i in err_list)/len(err_list))
        print(0.07,sum(i<=0.07 for i in err_list),sum(i<=0.07 for i in err_list)/len(err_list))
        print(0.08,sum(i<=0.08 for i in err_list),sum(i<=0.08 for i in err_list)/len(err_list))
        print(0.09,sum(i<=0.09 for i in err_list),sum(i<=0.09 for i in err_list)/len(err_list))
        print(0.1,sum(i<=0.1 for i in err_list),sum(i<=0.1 for i in err_list)/len(err_list))
        ll=sum(i>0.1 for i in err_list)
        print(ll,ll/len(err_list))


        print(sum(err_list)/1003)





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
