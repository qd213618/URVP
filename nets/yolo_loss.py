import torch
import torch.nn as nn
import numpy as np
import math
from scipy.optimize import leastsq

from common.utils import bbox_iou

class YOLOLoss_hrnet(nn.Module):
    def __init__(self, anchors, lambda_xy, img_size):
        super(YOLOLoss_hrnet, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = 1
        # self.slice_num = slice_num
        self.bbox_attrs = 3#13   org is without *2
        self.img_size = img_size
        self.ignore_threshold = 0.5
        self.lambda_xy = lambda_xy #org is 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 5.0
        # self.lambda_cls = 0
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None):
        # print(input.shape,targets.shape)
        bs = input.size(0)# batch size
        in_h = input.size(2)
        in_w = input.size(3)

        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])          # Center x
        y = torch.sigmoid(prediction[..., 1])          # Center y
        conf = torch.sigmoid(prediction[..., 2])       # Conf
        if targets is not None:
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,
                                                                           in_w, in_h,
                                                                           self.ignore_threshold)
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty = tx.cuda(), ty.cuda()
            tconf = tconf.cuda()
            loss_x = self.bce_loss(x * mask, tx * mask)#org is bce_loss
            loss_y = self.bce_loss(y * mask, ty * mask)#org is bce_loss
            loss_conf = self.bce_loss(conf * mask, mask) + 0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)

            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                loss_conf * self.lambda_conf

            return loss#, loss_x.item(), loss_y.item()#, loss_conf.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :2].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            # Results
            _scale = torch.Tensor([stride_w, stride_h]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 2) * _scale, conf.view(bs, -1, 1)), -1)
            return output.data

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)
        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            # Convert to position relative to box
            gx = target[b, 0, 1] * in_w
            gy = target[b, 0, 2] * in_h
            gw = target[b, 0, 3] * in_w
            gh = target[b, 0, 4] * in_h
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                              np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
            best_n = np.argmax(anch_ious)
            # Masks
            mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
            # object
            tconf[b, best_n, gj, gi] = 1
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, 0, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls
