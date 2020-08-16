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
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tensorboardX import SummaryWriter

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain_PL
from nets.yolo_loss import YOLOLoss_hrnet
from common.coco_dataset_bp_yolo import COCODataset
from common.evaluation import accuracy, AverageMeter, final_preds


class Trainer():
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG,
                            format="[%(asctime)s %(filename)s] %(message)s")

        if len(sys.argv) != 2:
            logging.error("Usage: python training.py params.py")
            sys.exit()
        params_path = sys.argv[1]
        if not os.path.isfile(params_path):
            logging.error("no params file found! path: {}".format(params_path))
            sys.exit()
        self.config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
        self.config["global_step"] = self.config.get("start_step", 0)
        self.config["batch_size"] *= len(self.config["parallels"])
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, self.config["parallels"]))
        # DataLoader
        self.dataloader = torch.utils.data.DataLoader(COCODataset(self.config["train_path"],
                                                             (self.config["img_w"], self.config["img_h"]),
                                                             is_training=True),
                                                 batch_size=self.config["batch_size"],
                                                 shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
        self.val_dataloader = torch.utils.data.DataLoader(COCODataset(self.config["test_path"],
                                                             (self.config["img_w"], self.config["img_h"]),
                                                             is_training=False),
                                                 batch_size=50,
                                                 shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    def train(self, config):
        c1max=0
        epoch_max=0
        epoch_th=20
        lr = 1e-3
        print("lamda_xy = ",config["lambda_xy"])
        # loss_min=1.5*(config["lambda_xy"]/175)
        # print(loss_min)
        is_training = False if config.get("export_onnx") else True
        net = ModelMain_PL(config, is_training=is_training)
        net.train(is_training)
        optimizer = self._get_optimizer(config, net)
        # Set data parallel
        net = nn.DataParallel(net)
        net = net.cuda()
        loss = nn.MSELoss()
        loss_coor = YOLOLoss_hrnet(config["yolo"]["anchors"][0], config["lambda_xy"], (config["img_w"], config["img_h"]))
        # Restore pretrain model
        if config["pretrain_snapshot"]:
            logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
            state_dict = torch.load(config["pretrain_snapshot"])
            net.load_state_dict(state_dict)
        # Start the training loop
        logging.info("Start training.")

        for epoch in range(config["epochs"]):
            if epoch > epoch_th + 10 and epoch > 10:
                lr = lr/10.0
                epoch_th = epoch + 1

            self.adjust_learning_rate(optimizer, lr)

            for step, samples in enumerate(self.dataloader):
                net.train(is_training)
                self.images, self.labels = samples["image"], samples["label"]
                self.labels_h = samples['label_h']
                # print(samples)
                self.labels_coor = samples['label_coor']
                # print(self.labels_coor)
                self.step = step
                self.labels= self.labels.cuda()
                self.labels_coor= self.labels_coor.cuda()
                self.labels_h= self.labels_h.cuda()

                start_time = time.time()
                config["global_step"] += 1

                # Forward and backward
                optimizer.zero_grad()
                hm, outh, coord = net(self.images)
                # hm = net(self.images)
                # print(len(output))
                losses_name = ["total_loss"]
                losses = []

                for _ in range(len(losses_name)):
                    losses.append([])

                cur_loss = loss(hm, self.labels)
                cur_lossh = loss(outh, self.labels_h)
                cur_loss_coor = loss_coor(coord, self.labels_coor)

                losses[0].append(cur_loss + cur_loss_coor + cur_lossh)
                losses = [sum(l) for l in losses]
                lossa = losses[0]

                lossa.backward()
                optimizer.step()

                # print(loss.shape)

                if self.step > 0 and self.step % 10 == 0:#org to be 10
                    _loss = lossa.item()
                    duration = float(time.time() - start_time)
                    example_per_second = config["batch_size"] / duration
                    lr = optimizer.param_groups[0]['lr']
                    logging.info(
                        "epoch [%.3d] iter = %d loss = %.2f example/sec = %.3f lr = %.5f"%
                        (epoch, self.step, _loss, example_per_second, lr)
                    )
                    config["tensorboard_writer"].add_scalar("lr",
                                                            lr,
                                                            config["global_step"])
                    config["tensorboard_writer"].add_scalar("example/sec",
                                                            example_per_second,
                                                            config["global_step"])
                    # loss_epoch += _loss
                    config["tensorboard_writer"].add_scalar("loss",
                                                            _loss,
                                                            config["global_step"])
            # for val
            if epoch > 9 and epoch % 1 == 0:
                net.eval()
                Drms1=[]
                Drms2=[]
                Drms=[]
                err_list=[]
                for step, samples in enumerate(self.val_dataloader):
                    images, labels = samples["image"], samples["label"]
                    labels_h = samples["label_h"]
                    labels = labels.cuda()
                    labels_h = labels_h.cuda()
                    labels_coor = samples["label_coor"]
                    # labels_coor = labels_coor.cuda()
                    with torch.no_grad():
                        # pred = net(images)

                        pred, predh, coord = net(images)
                        # print(coord.shape)
                        # for coord
                        det = loss_coor(coord)

                        for id, detections in enumerate(det):
                            _,indextensor=detections.max(0)
                            # print(indextensor.shape)
                            detections=detections[indextensor[2]].unsqueeze(0)
                            # print(detections.shape)

                            for x1, y1, conf in detections:
                                ori_h, ori_w = images[id].shape[1:3]
                                org_x = labels_coor[id][0][1]*config["img_h"]
                                org_y = labels_coor[id][0][2]*config["img_h"]

                                err = math.sqrt(pow(float(org_x - x1),2)+pow(float(org_y - y1),2)) / (256.*math.sqrt(2))
                                err_list.append(err)
                c1 = sum(i<=0.01 for i in err_list)
                cc = [sum(i<=0.01 for i in err_list),sum(i<=0.02 for i in err_list),
                        sum(i<=0.03 for i in err_list),sum(i<=0.04 for i in err_list),
                        sum(i<=0.05 for i in err_list),sum(i<=0.06 for i in err_list),
                        sum(i<=0.07 for i in err_list),sum(i<=0.08 for i in err_list),
                        sum(i<=0.09 for i in err_list),sum(i>0.1 for i in err_list)]
                print(c1,c1max)

                print(cc,len(err_list))

                if c1max < c1:
                    c1max = c1
                    epoch_max = epoch
                    epoch_th = epoch
                    self._save_checkpoint(net.state_dict(), config, "model_max_%d.pth" % (config["lambda_xy"]) )
                    print(cc,len(err_list))
                    print([sum(i<=0.01 for i in err_list)/len(err_list),sum(i<=0.02 for i in err_list)/len(err_list),
                            sum(i<=0.03 for i in err_list)/len(err_list),sum(i<=0.04 for i in err_list)/len(err_list),
                            sum(i<=0.05 for i in err_list)/len(err_list),sum(i<=0.06 for i in err_list)/len(err_list),
                            sum(i<=0.07 for i in err_list)/len(err_list),sum(i<=0.08 for i in err_list)/len(err_list),
                            sum(i<=0.09 for i in err_list)/len(err_list),sum(i>0.1 for i in err_list)/len(err_list)])


        self._save_checkpoint(net.state_dict(), config, "model_%d_%d_xy%d.pth" % (epoch_max,c1max,config["lambda_xy"]))
        logging.info("Bye~")

    # best_eval_result = 0.0
    def _save_checkpoint(self, state_dict, config, savename, evaluate_func=None):
        # global best_eval_result
        checkpoint_path = os.path.join(config["sub_working_dir"], savename)
        torch.save(state_dict, checkpoint_path)
        logging.info("Model checkpoint saved to %s" % checkpoint_path)


    def _get_optimizer(self, config, net):
        optimizer = None

        # Assign different lr for each layer
        params = None
        base_params = list(
            map(id, net.backbone.parameters())
        )
        logits_params = filter(lambda p: id(p) not in base_params, net.parameters())

        if not config["lr"]["freeze_backbone"]:
            params = [
                {"params": logits_params, "lr": config["lr"]["other_lr"]},
                {"params": net.backbone.parameters(), "lr": config["lr"]["backbone_lr"]},
            ]
        else:
            logging.info("freeze backbone's parameters.")
            for p in net.backbone.parameters():
                p.requires_grad = False
            params = [
                {"params": logits_params, "lr": config["lr"]["other_lr"]},
            ]

        # Initialize optimizer class
        if config["optimizer"]["type"] == "adam":
            optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"])
        elif config["optimizer"]["type"] == "amsgrad":
            optimizer = optim.Adam(params, weight_decay=config["optimizer"]["weight_decay"],
                                   amsgrad=True)
        elif config["optimizer"]["type"] == "rmsprop":
            optimizer = optim.RMSprop(params, weight_decay=config["optimizer"]["weight_decay"])
        else:
            # Default to sgd
            logging.info("Using SGD optimizer.")
            optimizer = optim.SGD(params, momentum=0.9,
                                  weight_decay=config["optimizer"]["weight_decay"],
                                  nesterov=(config["optimizer"]["type"] == "nesterov"))

        return optimizer
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def main(self):
        # logging.basicConfig(level=logging.DEBUG,
        #                     format="[%(asctime)s %(filename)s] %(message)s")
        #
        # if len(sys.argv) != 2:
        #     logging.error("Usage: python training.py params.py")
        #     sys.exit()
        # params_path = sys.argv[1]
        # if not os.path.isfile(params_path):
        #     logging.error("no params file found! path: {}".format(params_path))
        #     sys.exit()
        # config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
        # self.config["batch_size"] *= len(self.config["parallels"])

        # Create sub_working_dir
        # for i in range(self.config['lambda_xy_start'], self.config['lambda_xy_end'], self.config['lambda_xy_step']):
        for i in self.config['lambda_list']:
            self.config['lambda_xy'] = i
            sub_working_dir = '{}/{}/size{}x{}_try{}/{}'.format(
                self.config['working_dir'], self.config['model_params']['backbone_name'],
                self.config['img_w'], self.config['img_h'], self.config['try'],
                time.strftime("%Y%m%d%H%M%S", time.localtime()))
            if not os.path.exists(sub_working_dir):
                os.makedirs(sub_working_dir)
            self.config["sub_working_dir"] = sub_working_dir
            logging.info("sub working dir: %s" % sub_working_dir)

            # Creat tf_summary writer
            self.config["tensorboard_writer"] = SummaryWriter(sub_working_dir)
            logging.info("Please using 'python -m tensorboard.main --logdir={}'".format(sub_working_dir))

            # Start training
            # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, self.config["parallels"]))
            self.train(self.config)

if __name__ == "__main__":
    # main()
    agent = Trainer()
    agent.main()
