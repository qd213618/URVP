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
# sys.path.insert(0, os.path.join(MY_DIRNAME, '..', 'evaluate'))
# from nets.backbone import backbone_fn
from nets.model_main import ModelMain_PL
from nets.yolo_loss import YOLOLoss_hrnet
# from common.coco_dataset_yolo import COCODataset
from common.coco_dataset_bp_yolo import COCODataset
from common.evaluation import accuracy, AverageMeter, final_preds
# from nets.backbone.sync_batchnorm.replicate import DataParallelWithCallback, patch_replication_callback

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
        # is_training = False if self.config.get("export_onnx") else True
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
        # self.prefetcher = data_prefetcher(self.dataloader)
        # self.images, self.labels = self.prefetcher.next()

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
        # print(net.shape())
        optimizer = self._get_optimizer(config, net)
        # Set data parallel
        net = nn.DataParallel(net)
        # if config["yolo"]["batchnorm"]=="SYBN":
        #     patch_replication_callback(net)
        net = net.cuda()
        loss = nn.MSELoss()
        loss_coor = YOLOLoss_hrnet(config["yolo"]["anchors"][0], config["lambda_xy"], (config["img_w"], config["img_h"]))
        # idx = [1]
        # yolo_losses = []
        # for i in range(3):
        #     yolo_losses.append(YOLOLoss_line_pp(config["yolo"]["anchors"][i],
        #                                 config["yolo"]["classes"], config["yolo"]["line_num"], config["lambda_xy"], (config["img_w"], config["img_h"])))

        # Restore pretrain model
        if config["pretrain_snapshot"]:
            logging.info("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
            state_dict = torch.load(config["pretrain_snapshot"])
            net.load_state_dict(state_dict)
        # Start the training loop
        logging.info("Start training.")
        # if config["train_path"]=="/home/lyb/datasets/vp/finalpoint_line.txt":
        #     loss_epoch_min = (loss_min+0.1)*5
        # else:
        #     loss_epoch_min = (loss_min+0.1)*30
        # loss_epoch_min = (loss_min+0.1)*31 #int(4352 / (20*config["batch_size"]))
        # loss_epoch = loss_epoch_min+1
        # print("avg_min:",loss_epoch_min,self.dataloader.__len__())

        for epoch in range(config["epochs"]):
            # acces = AverageMeter()
            # lr = 1e-3 #for hg 75epoch/down
            # ##################################
            if epoch > epoch_th + 10 and epoch > 10:
            # if epoch > epoch_th + 10:
                lr = lr/10.0
                epoch_th = epoch + 1
            #  ##########################################
            # if epoch > 10:
            #     lr = 0.0001
                # without rote
                # for sharp_peleenet with pretrain >15 down to 0.001 with 60 epochs
                # for mobilev2 without pretrain >25 down to 0.001 with 70 epochs
                # for mobilev3 without pretrain >25 down to 0.001 with 70 epochs
                # with rote
                # for mobilev2,v3,pelee with pretrain >10 down to 0.001 with 50 epochs
                # for mobilev2 without pretrain >15 down to 0.001 with 50 epochs
                # for finetune lr=0.001, epoch >10 down to lr=0.00001
            # if epoch > 60: #for rote change to 10 when 0.001 org is 120/160
            #     lr = 1e-4
            # if epoch > 100: #for rote change to 10 when 0.001
            #     lr = 1e-5
            # if epoch > 140: #for rote change to 10 when 0.001
            #     lr = 1e-6
            # if epoch > 100:
            #     lr = 0.0001
            # if epoch > 37:
            #     lr = 0.0001
            self.adjust_learning_rate(optimizer, lr)
            # if loss_epoch_min > loss_epoch:
            #     loss_epoch_min = loss_epoch
            #     _save_checkpoint(net.state_dict(), config, "model_min_avg.pth")
            # loss_epoch = 0
            # self.images, self.labels = self.prefetcher.next()
            # self.step = 0
            # while self.images is not None:
            # for training
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
                # for i in range(2):
                # _loss_item = yolo_losses[0](outputs, self.labels)
                # for j, l in enumerate(_loss_item):
                #     # print(l.shape)
                #     losses[j].append(l)
                # for i in range(3):
                #     _loss_item = yolo_losses[i](outputs[i], self.labels)
                #     for j, l in enumerate(_loss_item):
                #         # print(l.shape)
                #         losses[j].append(l)
                # print(output.shape())
                cur_loss = loss(hm, self.labels)
                cur_lossh = loss(outh, self.labels_h)
                cur_loss_coor = loss_coor(coord, self.labels_coor)
                #
                # print(cur_loss,cur_loss_coor)
                # score_map = output[-1].cpu() if type(output) == list else output.cpu()
                # acc = accuracy(score_map, self.labels.cpu(), idx)
                # acces.update(acc[0], self.images.size(0))
                #
                # cur_loss = loss(output, self.labels)#for pl4vp
                #
                # cur_loss = 0
                # for o in output:#for heatmap
                #     cur_loss += loss(o,self.labels)

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
                    # if _loss<10:
                    # if loss_min>_loss:
                    #     loss_min=_loss
                    #     self._save_checkpoint(net.state_dict(), config, "model_min.pth")
                    # if config["train_path"]=="/home/lyb/datasets/vp/finalpoint_line_rote.txt":
                        # for rote epoch >23, for finetune epoch > -1
                        # if epoch > 23:
                        #     if self.step % 50 == 0:
                        #         self._save_checkpoint(net.state_dict(), config, "model_e%d_s%d_xy%d_l%d.pth" % (epoch, self.step, config["lambda_xy"], config["yolo"]['line_num']))

                    # for i, name in enumerate(losses_name):
                    #     value = _loss if i == 0 else losses[i]
                    #     config["tensorboard_writer"].add_scalar(name,
                    #                                             value,
                    #                                             config["global_step"])
                # self.images, self.labels = self.prefetcher.next()
                # self.step += 1

            #for rote epoch >15, for finetune epoch >-1
            # acces.reset()
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
                                # print(x1, y1)
                                ori_h, ori_w = images[id].shape[1:3]
                                # # print(ori_h,ori_w)
                                # pre_h, pre_w = config["img_h"], config["img_w"]
                                # y1 = (y1 / pre_h) * ori_h
                                # x1 = (x1 / pre_w) * ori_w

                                org_x = labels_coor[id][0][1]*config["img_h"]
                                org_y = labels_coor[id][0][2]*config["img_h"]

                #                 lock=0
                #                 a=0.5*math.sqrt(pow((float(labels_coor[id][1][1])-float(labels_coor[id][1][3]))*ori_w,2)+pow((float(labels_coor[id][1][2])-float(labels_coor[id][1][4]))*ori_h,2))
                #                 c2=math.sqrt(pow((float(labels_coor[id][1][3])*ori_w-x1),2)+pow((float(labels_coor[id][1][4])*ori_h-y1),2))
                #                 c1=math.sqrt(pow((float(labels_coor[id][1][1])*ori_w-x1),2)+pow((float(labels_coor[id][1][2])*ori_h-y1),2))
                #                 b=math.sqrt(pow(((float(labels_coor[id][1][1])+float(labels_coor[id][1][3]))*ori_w/2-x1),2)+pow(((float(labels_coor[id][1][2])+float(labels_coor[id][1][4]))*ori_h/2-y1),2))
                #                 if c1>c2:
                #                     c=c2
                #                 else:
                #                     c=c1
                #                 s=(a+b+c)/2
                #                 # print(a,b,c,s)
                #                 try:
                #                     if s<max(a,max(b,c)):
                #                         s=max(a,max(b,c))
                #                     h=2*math.sqrt(s*(s-a)*(s-b)*(s-c))/b
                #                     ss=0
                #                     for i in range(int(a)):
                #                         ss+=pow(h*(a-i)/a,2)
                #                     avg1=math.sqrt(ss/(2*int(a)+1))
                #                     Drms1.append(avg1)
                #                     lock=1
                #                     # if avg1>=5:
                #                     #     print(images_path[batch_size*step+idx])
                #                 except Exception as e:
                #                     print(e)
                #
                #                 a=0.5*math.sqrt(pow((float(labels_coor[id][2][1])-float(labels_coor[id][2][3]))*ori_w,2)+pow((float(labels_coor[id][2][2])-float(labels_coor[id][2][4]))*ori_h,2))
                #                 c2=math.sqrt(pow((float(labels_coor[id][2][3])*ori_w-x1),2)+pow((float(labels_coor[id][2][4])*ori_h-y1),2))
                #                 c1=math.sqrt(pow((float(labels_coor[id][2][1])*ori_w-x1),2)+pow((float(labels_coor[id][2][2])*ori_h-y1),2))
                #                 b=math.sqrt(pow(((float(labels_coor[id][2][1])+float(labels_coor[id][2][3]))*ori_w/2-x1),2)+pow(((float(labels_coor[id][2][2])+float(labels_coor[id][2][4]))*ori_h/2-y1),2))
                #                 if c1>c2:
                #                     c=c2
                #                 else:
                #                     c=c1
                #                 s=(a+b+c)/2
                #                 # print(a,b,c,s)
                #                 try:
                #                     if s<max(a,max(b,c)):
                #                         s=max(a,max(b,c))
                #                     h=2*math.sqrt(s*(s-a)*(s-b)*(s-c))/b
                #                     ss=0
                #                     for i in range(int(a)):
                #                         ss+=pow(h*(a-i)/a,2)
                #                     avg2=math.sqrt(ss/(2*int(a)+1))
                #                     if lock==1:
                #                         Drms2.append(avg2)
                #                     # lock=0
                #                 except Exception as e:
                #                     print(e)
                # c1=0
                # c2=0
                # c3=0
                # c4=0
                # c5=0
                # c6=0
                # c7=0
                # c9=0
                # c8=0
                # c10=0
                # c10p=0
                # for d in range(len(Drms2)):
                #     Drms.append((Drms1[d]+Drms2[d])/2)
                #
                # for d in Drms:
                #     if d<=1:
                #         c1+=1
                #     if d<=2:
                #         c2+=1
                #     if d<=3:
                #         c3+=1
                #     if d<=4:
                #         c4+=1
                #     if d<=5:
                #         c5+=1
                #     if d<=6:
                #         c6+=1
                #     if d<=7:
                #         c7+=1
                #     if d<=8:
                #         c8+=1
                #     if d<=9:
                #         c9+=1
                #     if d<=10:
                #         c10+=1
                #     c10p+=1
                # logging.info('c1-10:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (c1/c10p,c2/c10p,c3/c10p,c4/c10p,c5/c10p,c6/c10p,c7/c10p,c8/c10p,c9/c10p,c10/c10p))
                # logging.info("c1-c10: %d %d %d %d %d %d %d %d %d %d %d" % (c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c10p))
                # if c1max < c1:
                #     c1max = c1
                #     epoch_max = epoch
                #     epoch_th = epoch
                #     self._save_checkpoint(net.state_dict(), config, "model_max_%d.pth" % (config["lambda_xy"]) )
                #     print(c1,c1max)
                                ########################for kong test############
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
                # print([sum(i<=0.01 for i in err_list)/len(err_list),sum(i<=0.02 for i in err_list)/len(err_list),
                #         sum(i<=0.03 for i in err_list)/len(err_list),sum(i<=0.04 for i in err_list)/len(err_list),
                #         sum(i<=0.05 for i in err_list)/len(err_list),sum(i<=0.06 for i in err_list)/len(err_list),
                #         sum(i<=0.07 for i in err_list)/len(err_list),sum(i<=0.08 for i in err_list)/len(err_list),
                #         sum(i<=0.09 for i in err_list)/len(err_list),sum(i>0.1 for i in err_list)/len(err_list)])

                if c1max < c1:
                    c1max = c1
                    epoch_max = epoch
                    epoch_th = epoch
                    self._save_checkpoint(net.state_dict(), config, "model_max_%d.pth" % (config["lambda_xy"]) )
                    # print(c1,c1max,c10,len(err_list))
                    # print(c1,c1max)
                    print(cc,len(err_list))
                    print([sum(i<=0.01 for i in err_list)/len(err_list),sum(i<=0.02 for i in err_list)/len(err_list),
                            sum(i<=0.03 for i in err_list)/len(err_list),sum(i<=0.04 for i in err_list)/len(err_list),
                            sum(i<=0.05 for i in err_list)/len(err_list),sum(i<=0.06 for i in err_list)/len(err_list),
                            sum(i<=0.07 for i in err_list)/len(err_list),sum(i<=0.08 for i in err_list)/len(err_list),
                            sum(i<=0.09 for i in err_list)/len(err_list),sum(i>0.1 for i in err_list)/len(err_list)])



                #
                #
                #
            # if epoch > 50 and epoch % 5 == 0:#for rote change to 15 no rote to 26
            #     # net.train(False)
            #     self._save_checkpoint(net.state_dict(), config, "model_%d_%d_l%d.pth" % (epoch,config["lambda_xy"],config["yolo"]['line_num']))
                    # net.train(True)

            # lr_scheduler.step()

        # net.train(False)
        # self._save_checkpoint(net.state_dict(), config, "model.pth")
        self._save_checkpoint(net.state_dict(), config, "model_%d_%d_xy%d.pth" % (epoch_max,c1max,config["lambda_xy"]))
        # net.train(True)
        logging.info("Bye~")

    # best_eval_result = 0.0
    def _save_checkpoint(self, state_dict, config, savename, evaluate_func=None):
        # global best_eval_result
        checkpoint_path = os.path.join(config["sub_working_dir"], savename)
        torch.save(state_dict, checkpoint_path)
        logging.info("Model checkpoint saved to %s" % checkpoint_path)
        # eval_result = evaluate_func(config)
        # if eval_result > best_eval_result:
            # best_eval_result = eval_result
            # logging.info("New best result: {}".format(best_eval_result))
            # best_checkpoint_path = os.path.join(config["sub_working_dir"], 'model_best.pth')
            # shutil.copyfile(checkpoint_path, best_checkpoint_path)
            # logging.info("Best checkpoint saved to {}".format(best_checkpoint_path))
        # else:
            # logging.info("Best result: {}".format(best_eval_result))


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
